#!/usr/bin/env python3
# -- coding: utf-8 --

"""CLI report output and untrusted filename handling."""

# Imports
import csv
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from click.testing import CliRunner

from eyeballer import cli as CLI
LABELS = ["custom404", "login", "webapp", "oldlooking", "parked"]


class CLITest(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shot = self.root / 'image "quoted" & spaced.png'
        self.shot.touch()
        self.weights = self.root / "weights.h5"
        self.weights.touch()
        self.output = self.root / "reports"
        self.model = Mock()
        self.model.predict.return_value = [{"filename": str(self.shot), **dict.fromkeys(LABELS, 0.1)}]
        self.factory = Mock(return_value=self.model)
        modules = patch.dict(sys.modules, {"eyeballer.model": SimpleNamespace(EyeballModel=self.factory, DATA_LABELS=LABELS)})
        modules.start()
        self.addCleanup(modules.stop)

    def invoke(self, *extra):
        return self.runner.invoke(CLI.cli, ["--weights", str(self.weights), "predict", str(self.shot),
                                             "--output-dir", str(self.output), *extra])

    def test_single_image_reports_from_another_working_directory(self):
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(elsewhere)
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        with (self.output / "results.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["filename"], str(self.shot))
        html = (self.output / "results.html").read_text()
        self.assertIn("Unclassified", html)
        self.assertIn("%22quoted%22", html)
        self.assertIn("link.textContent", html)
        self.assertNotIn('var results = {\'', html)
        for label in LABELS:
            self.assertIn(f'data-tag="{label}" aria-pressed="false"', html)

    def gowitness_db(self, rows):
        database = self.root / "gowitness.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE results (id INTEGER PRIMARY KEY, filename TEXT, title TEXT, probed_at TEXT)")
            connection.executemany("INSERT INTO results (filename, title, probed_at) VALUES (?, ?, ?)", rows)
        connection.close()
        return database

    def test_gowitness_titles_add_device_filters(self):
        database = self.gowitness_db([(self.shot.name, "Login - VMware ESXi", "2026-01-01"),
                                      (self.shot.name, "HP Device Status", "2026-01-02"),
                                      ("", "Unrelated", "2026-01-03")])
        result = self.invoke("--gowitness", str(database))
        self.assertEqual(result.exit_code, 0, result.output)
        html = (self.output / "results.html").read_text()
        self.assertIn('{"device": "Printer", "title": "HP Device Status"}', html)
        self.assertNotIn("VMware", html)

    def test_unreadable_gowitness_database_fails_before_model_construction(self):
        database = self.root / "not-a-database.sqlite3"
        database.write_text("not sqlite")
        result = self.invoke("--gowitness", str(database))
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Cannot read gowitness results", result.output)
        self.factory.assert_not_called()

    def test_device_type_first_match_wins(self):
        self.assertEqual(CLI.device_type("Machine Identification - BP-70M75"), "Printer")
        self.assertEqual(CLI.device_type("VMware Horizon"), "Remote access")
        self.assertEqual(CLI.device_type("Login - VMware ESXi"), "Hypervisor")
        self.assertEqual(CLI.device_type("CP-8841-3PCC Configuration Utility"), "IP phone")
        self.assertIsNone(CLI.device_type("IIS Windows Server"))

    def test_report_without_gowitness_has_no_pages(self):
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("var pages = {};", (self.output / "results.html").read_text())

    def test_multiple_images_preserve_scores(self):
        self.model.predict.return_value *= 2
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        with (self.output / "results.csv").open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_empty_results_fail_without_overwriting_reports(self):
        self.output.mkdir()
        old = self.output / "results.csv"
        old.write_text("existing evidence")
        self.model.predict.return_value = []
        result = self.invoke()
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No readable screenshots", result.output)
        self.assertEqual(old.read_text(), "existing evidence")

    def test_missing_weights_fail_before_model_construction(self):
        self.weights.unlink()
        result = self.invoke()
        self.assertNotEqual(result.exit_code, 0)
        self.factory.assert_not_called()

    def test_predict_defaults_to_bundled_weights(self):
        with patch.object(CLI, "DEFAULT_WEIGHTS", self.weights):
            result = self.runner.invoke(CLI.cli, ["predict", str(self.shot), "--output-dir", str(self.output)])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.factory.call_args.kwargs["weights_file"], str(self.weights))

    def test_missing_bundled_weights_fail_before_model_construction(self):
        with patch.object(CLI, "DEFAULT_WEIGHTS", self.root / "absent.h5"):
            result = self.runner.invoke(CLI.cli, ["predict", str(self.shot)])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("absent.h5 not found", result.output)
        self.factory.assert_not_called()

    def native_stderr(self, message, fail=False):
        sink = self.root / "stderr"
        saved = os.dup(2)
        try:
            with sink.open("wb") as stream:
                os.dup2(stream.fileno(), 2)
            try:
                with CLI.quiet_native_stderr():
                    os.write(2, message)
                    if fail:
                        raise RuntimeError
            except RuntimeError:
                pass
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        return sink.read_bytes()

    def test_native_startup_logs_hidden_unless_loading_fails(self):
        with patch.dict(os.environ):
            os.environ.pop("TF_CPP_MIN_LOG_LEVEL", None)
            self.assertEqual(self.native_stderr(b"cuda probe\n"), b"")
            self.assertEqual(self.native_stderr(b"load failure\n", fail=True), b"load failure\n")

    def test_tf_log_level_disables_native_capture(self):
        with patch.dict(os.environ, {"TF_CPP_MIN_LOG_LEVEL": "0"}):
            self.assertEqual(self.native_stderr(b"cuda probe\n"), b"cuda probe\n")

    def test_help_does_not_load_model(self):
        result = self.runner.invoke(CLI.cli, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.factory.assert_not_called()

    def test_invalid_threshold_and_directory_heatmap(self):
        for threshold in ("1.1", "nan", "inf", "-0.1"):
            self.assertNotEqual(self.invoke("--threshold", threshold).exit_code, 0)
        result = self.runner.invoke(CLI.cli, ["--weights", str(self.weights), "predict", str(self.root), "--heatmap"])
        self.assertNotEqual(result.exit_code, 0)
        self.factory.assert_not_called()

    def test_script_tag_cannot_escape_report_data(self):
        output = self.root / "hostile.html"
        CLI.buildHTML({'</script><script>alert(1)</script>': []}, output)
        self.assertNotIn('</script><script>alert(1)</script>', output.read_text())


if __name__ == "__main__":
    unittest.main()
