#!/usr/bin/env python3
# -- coding: utf-8 --

# Imports
import click
from contextlib import closing, contextmanager
import csv
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

from jinja2 import Template

# Installers place the released v3 weights here so predict works without --weights.
DEFAULT_WEIGHTS = Path(__file__).with_name("bishop-fox-pretrained-v3.h5")

# Device types matched against gowitness page titles. The first match wins, so
# remote access comes before hypervisors to keep VMware Horizon out of the latter.
DEVICE_TYPES = [
    ("Printer", r"Machine Identification|Device Status|LaserJet|OfficeJet|PageWide|Xerox|Ricoh|Kyocera|Lexmark"
                r"|Konica|bizhub|Brother|Canon|Epson|Sharp|e-STUDIO|Printer|Web Image Monitor"),
    ("IP phone", r"\bCP-\d{4}|Cisco IP Phone|Polycom|Yealink|Grandstream|Avaya|Mitel"),
    ("Remote access", r"Horizon|Unified Access Gateway|Citrix|NetScaler|Teradici|PCoIP|Inuvika|VNC"
                      r"|Remote Desktop|RD Web|Guacamole|GlobalProtect"),
    ("Hypervisor", r"VMware|ESXi|vSphere|vCenter|Proxmox|Hyper-V|XenServer|XCP-ng|Nutanix|oVirt"),
    ("Out-of-band management", r"iDRAC|\biLO\b|Integrated Lights-Out|IPMI|Supermicro|XClarity|\bCIMC\b"
                               r"|Integrated Management Module"),
    ("Storage and backup", r"Unisphere|QNAP|Synology|DiskStation|NetApp|ONTAP|RecoverPoint|PowerStore"
                           r"|PowerVault|\bME\d{4}\b|Isilon|PowerScale|TrueNAS|Data Domain|Pure Storage|Nimble"
                           r"|3PAR|Veeam"),
    ("Network gear", r"Cisco|Nexus|Fortinet|FortiGate|Palo Alto|PAN-OS|Juniper|Aruba|Meraki|SonicWall|UniFi"
                     r"|Ubiquiti|MikroTik|pfSense|OPNsense|Check Point|BIG-IP"),
    ("Camera", r"Hikvision|Dahua|\bAxis\b|Network Camera|IP Camera|\bNVR\b|\bDVR\b|Avigilon"),
    ("Power", r"\bAPC\b|Network Management Card|PowerChute|Eaton|\bUPS\b|\bPDU\b|Vertiv|Liebert|Raritan"),
]
DEVICE_PATTERNS = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in DEVICE_TYPES]


@contextmanager
def quiet_native_stderr():
    """Hide TensorFlow's native startup logs, replaying them if loading fails.

    TensorFlow writes CUDA probing and oneDNN notices to file descriptor 2
    before its logger honours TF_CPP_MIN_LOG_LEVEL, so a level alone cannot
    silence them. Setting TF_CPP_MIN_LOG_LEVEL disables this capture.
    """
    if "TF_CPP_MIN_LOG_LEVEL" in os.environ:
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    try:
        with tempfile.TemporaryFile() as log:
            os.dup2(log.fileno(), 2)
            try:
                yield
            except BaseException:
                sys.stderr.flush()
                os.dup2(saved, 2)
                log.seek(0)
                with os.fdopen(os.dup(2), "wb") as stream:
                    stream.write(log.read())
                raise
            finally:
                sys.stderr.flush()
                os.dup2(saved, 2)
    finally:
        os.close(saved)


def load_model(model_kwargs):
    with quiet_native_stderr():
        from eyeballer.model import EyeballModel
        return EyeballModel(**model_kwargs)


@click.group(invoke_without_command=True)
@click.option('--weights', default=None, type=click.Path(), help="Weights file for input/output; predict defaults to the bundled v3 weights")
@click.option('--summary/--no-summary', default=False, help="Print model summary at start")
@click.option('--seed', default=None, type=int, help="RNG seed for data shuffling and transformations, defaults to random value")
@click.pass_context
def cli(ctx, weights, summary, seed):
    model_kwargs = {"weights_file": weights,
                    "print_summary": summary,
                    "seed": seed}

    #  pass the model to subcommands
    ctx.ensure_object(dict)
    # We only pass the kwargs so we can be lazy and make the model later after the subcommand cli is parsed. This
    # way, the user doesn't have to wait for tensorflow if they are just calling --help on a subcommand.
    ctx.obj['model_kwargs'] = model_kwargs


@cli.command()
@click.option('--graphs/--no-graphs', default=False, help="Save accuracy and loss graphs to file")
@click.option('--epochs', default=20, type=int, help="Number of epochs")  # TODO better help string
@click.option('--batchsize', default=32, type=int, help="Batch size")  # TODO better help string
@click.pass_context
def train(ctx, graphs, batchsize, epochs):
    model = load_model(ctx.obj['model_kwargs'])
    model.train(print_graphs=graphs, batch_size=batchsize, epochs=epochs)


@cli.command()
@click.argument('screenshot', type=click.Path(exists=True, path_type=Path))
@click.option('--heatmap', default=False, is_flag=True, help="Create a heatmap graphfor the prediction")
@click.option('--threshold', default=.5, type=click.FloatRange(0, 1), help="Threshold confidence for labeling")
@click.option('--output-dir', default='.', type=click.Path(file_okay=False, path_type=Path), help="Directory for results.csv and results.html")
@click.option('--gowitness', default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="gowitness SQLite database; adds page titles and device type filters to results.html")
@click.pass_context
def predict(ctx, screenshot, heatmap, threshold, output_dir, gowitness):
    if not math.isfinite(threshold):
        raise click.BadParameter("must be a finite value between 0 and 1", param_hint="--threshold")
    weights = Path(ctx.obj['model_kwargs']['weights_file'] or DEFAULT_WEIGHTS)
    if not weights.is_file():
        raise click.ClickException(f"Prediction requires a pretrained weights file; {weights} not found")
    ctx.obj['model_kwargs']['weights_file'] = str(weights)
    if heatmap and not screenshot.is_file():
        raise click.ClickException("--heatmap requires a single image")
    titles = load_gowitness_titles(gowitness) if gowitness else None
    model = load_model(ctx.obj['model_kwargs'])
    from eyeballer.model import DATA_LABELS
    results = model.predict(str(screenshot.resolve()))

    if not results:
        raise click.ClickException("No readable screenshots found; no reports written")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as csvfile:
        labelwriter = csv.DictWriter(csvfile, fieldnames=["filename", *DATA_LABELS])
        labelwriter.writeheader()
        labelwriter.writerows(results)
    pages = describe_pages(results, titles) if titles is not None else {}
    buildHTML(processResults(results, threshold), output_dir / "results.html", pages)
    click.echo(f"Output written to {output_dir / 'results.csv'} and {output_dir / 'results.html'}")

    if heatmap:
        from eyeballer.visualization import HeatMap
        # Generate a heatmap
        HeatMap(str(screenshot), model, threshold).generate(str(output_dir / "heatmap.png"))


def processResults(results, threshold):
    '''Filter the initial results dictionary and reformat it for use in JS.

        Keyword arguments:
        results -- dictionary output from predict function

    '''
    jsResults = {}

    for result in results:
        positiveTags = []

        for label, label_info in result.items():
            if (label == 'filename'):
                pass
            elif label_info > threshold:
                positiveTags.append(label)

        jsResults[Path(result['filename']).resolve().as_uri()] = positiveTags
    return(jsResults)


def load_gowitness_titles(database):
    '''Map screenshot file names to page titles from a gowitness SQLite database.

        When a file was captured more than once, the latest probe wins.
    '''
    try:
        with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            rows = connection.execute(
                "SELECT filename, title FROM results WHERE filename <> '' ORDER BY probed_at, id").fetchall()
    except sqlite3.Error as error:
        raise click.ClickException(f"Cannot read gowitness results from {database}: {error}")
    return {filename: title or "" for filename, title in rows}


def device_type(title):
    '''Return the first device type whose pattern matches a page title, or None.'''
    for name, pattern in DEVICE_PATTERNS:
        if pattern.search(title):
            return name
    return None


def describe_pages(results, titles):
    '''Key gowitness page titles and device types by the report's screenshot URIs.

        Keyword arguments:
        results -- dictionary output from predict function
        titles -- screenshot file name to page title, from load_gowitness_titles
    '''
    pages = {}
    for result in results:
        path = Path(result['filename'])
        title = titles.get(path.name, "")
        pages[path.resolve().as_uri()] = {"title": title, "device": device_type(title)}
    return pages


def buildHTML(jsResults, output_file=Path('results.html'), pages=None):
    '''Build HTML around the JS Dictionary that is passed from processResults.

        Keyword arguments:
        jsResults -- dictionary output from processResults function
        pages -- optional dictionary output from describe_pages function
    '''
    html_output = ""
    with Path(__file__).with_name("prediction_output_template.html").open(encoding="utf-8") as template_file:
        template = Template(template_file.read())
        html_output = template.render(jsResults=jsResults, pages=pages or {},
                                      deviceTypes=[name for name, _ in DEVICE_TYPES])

    with open(output_file, 'w', encoding="utf-8") as file:
        file.write(html_output)


def pretty_print_evaluation(results):
    """Print a human-readable summary of the evaluation"""
    from eyeballer.model import DATA_LABELS
    # We use 4.2% to handle all the way from "  0.00%" (7chars) to "100.00%" (7chars)
    for label in DATA_LABELS:
        print("{} Precision Score: {:4.2%}".format(label, results[label]['precision']))
        print("{} Recall Score: {:4.2%}".format(label, results[label]['recall']))
    print("'None of the above' Precision: {:4.2%}".format(results['none_of_the_above_precision']))
    print("'None of the above' Recall: {:4.2%}".format(results['none_of_the_above_recall']))
    print("All or nothing Accuracy: {:4.2%}".format(results['all_or_nothing_accuracy']))
    print("Overall Binary Accuracy: {:4.2%}".format(results['total_binary_accuracy']))
    print("Top 10 worst predictions: {}".format(results['top_10_worst'][1]))


@cli.command()
@click.option('--threshold', default=.5, type=float, help="Threshold confidence for labeling")
@click.pass_context
def evaluate(ctx, threshold):
    model = load_model(ctx.obj['model_kwargs'])
    results = model.evaluate(threshold)
    pretty_print_evaluation(results)


if __name__ == '__main__':
    cli()
