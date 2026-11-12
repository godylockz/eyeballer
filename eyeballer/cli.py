#!/usr/bin/env python3
# -- coding: utf-8 --

# Imports
import click
import csv
import math
from pathlib import Path

from jinja2 import Template

# Installers place the released v3 weights here so predict works without --weights.
DEFAULT_WEIGHTS = Path(__file__).with_name("bishop-fox-pretrained-v3.h5")


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
    from eyeballer.model import EyeballModel
    model = EyeballModel(**ctx.obj['model_kwargs'])
    model.train(print_graphs=graphs, batch_size=batchsize, epochs=epochs)


@cli.command()
@click.argument('screenshot', type=click.Path(exists=True, path_type=Path))
@click.option('--heatmap', default=False, is_flag=True, help="Create a heatmap graphfor the prediction")
@click.option('--threshold', default=.5, type=click.FloatRange(0, 1), help="Threshold confidence for labeling")
@click.option('--output-dir', default='.', type=click.Path(file_okay=False, path_type=Path), help="Directory for results.csv and results.html")
@click.pass_context
def predict(ctx, screenshot, heatmap, threshold, output_dir):
    if not math.isfinite(threshold):
        raise click.BadParameter("must be a finite value between 0 and 1", param_hint="--threshold")
    weights = Path(ctx.obj['model_kwargs']['weights_file'] or DEFAULT_WEIGHTS)
    if not weights.is_file():
        raise click.ClickException(f"Prediction requires a pretrained weights file; {weights} not found")
    ctx.obj['model_kwargs']['weights_file'] = str(weights)
    if heatmap and not screenshot.is_file():
        raise click.ClickException("--heatmap requires a single image")
    from eyeballer.model import EyeballModel, DATA_LABELS
    model = EyeballModel(**ctx.obj['model_kwargs'])
    results = model.predict(str(screenshot.resolve()))

    if not results:
        raise click.ClickException("No readable screenshots found; no reports written")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as csvfile:
        labelwriter = csv.DictWriter(csvfile, fieldnames=["filename", *DATA_LABELS])
        labelwriter.writeheader()
        labelwriter.writerows(results)
    buildHTML(processResults(results, threshold), output_dir / "results.html")
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


def buildHTML(jsResults, output_file=Path('results.html')):
    '''Build HTML around the JS Dictionary that is passed from processResults.

        Keyword arguments:
        jsResults -- dictionary output from processResults function
    '''
    html_output = ""
    with Path(__file__).with_name("prediction_output_template.html").open(encoding="utf-8") as template_file:
        template = Template(template_file.read())
        html_output = template.render(jsResults=jsResults)

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
    from eyeballer.model import EyeballModel
    model = EyeballModel(**ctx.obj['model_kwargs'])
    results = model.evaluate(threshold)
    pretty_print_evaluation(results)


if __name__ == '__main__':
    cli()
