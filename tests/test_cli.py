#!/usr/bin/env python3
# -- coding: utf-8 --

"""CLI report output and untrusted filename handling."""

# Imports
import csv
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from click.testing import CliRunner

CLI = runpy.run_path(str(Path(__file__).resolve().parents[1] / "eyeballer.py"))
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
        return self.runner.invoke(CLI["cli"], ["--weights", str(self.weights), "predict", str(self.shot),
                                             "--output-dir", str(self.output), *extra])

    def test_single_image_reports_from_another_working_directory(self):
        with self.runner.isolated_filesystem():
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

    def test_help_does_not_load_model(self):
        result = self.runner.invoke(CLI["cli"], ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.factory.assert_not_called()

    def test_invalid_threshold_and_directory_heatmap(self):
        for threshold in ("1.1", "nan", "inf", "-0.1"):
            self.assertNotEqual(self.invoke("--threshold", threshold).exit_code, 0)
        result = self.runner.invoke(CLI["cli"], ["--weights", str(self.weights), "predict", str(self.root), "--heatmap"])
        self.assertNotEqual(result.exit_code, 0)
        self.factory.assert_not_called()

    def test_script_tag_cannot_escape_report_data(self):
        output = self.root / "hostile.html"
        CLI["buildHTML"]({'</script><script>alert(1)</script>': []}, output)
        self.assertNotIn('</script><script>alert(1)</script>', output.read_text())


if __name__ == "__main__":
    unittest.main()
