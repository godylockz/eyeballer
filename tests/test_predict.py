#!/usr/bin/env python3
# -- coding: utf-8 --

"""Inference checks against the released five-label model."""

# Imports
import hashlib
import io
import math
import os
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class PredictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        weights = os.environ.get("EYEBALLER_TEST_WEIGHTS")
        if not weights:
            raise unittest.SkipTest("Set EYEBALLER_TEST_WEIGHTS to the v3 release weights")
        checksum = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
        if checksum != "dd1c3741f6141f096a5ca6ee882df74bd5cb5b907c2df4b5548c4b02bd42afcd":
            raise ValueError("Tests require the released v3 model")
        from eyeballer.model import EyeballModel
        import tensorflow as tf
        with patch.object(tf.keras.applications.mobilenet, "MobileNet", wraps=tf.keras.applications.mobilenet.MobileNet) as constructor:
            cls.model = EyeballModel(weights_file=weights, quiet=True)
        assert constructor.call_args.kwargs["weights"] is None
        cls.data = Path(__file__).resolve().parent / "data"

    def test_login_and_404(self):
        for name, label in (("login.png", "login"), ("404.png", "custom404")):
            with self.subTest(name=name):
                row = self.model.predict(str(self.data / name))[0]
                self.assertGreater(row[label], .5)

    def test_directory_outputs_five_finite_probabilities(self):
        rows = self.model.predict(str(self.data))
        self.assertEqual(len(rows), 7)
        for row in rows:
            self.assertEqual(set(row), {"filename", "custom404", "login", "webapp", "oldlooking", "parked"})
            for label, score in row.items():
                if label != "filename":
                    self.assertTrue(math.isfinite(float(score)) and 0 <= score <= 1)

    def test_invalid_images_and_missing_paths(self):
        for name in ("empty.png", "invalid.png"):
            self.assertEqual(self.model.predict(str(self.data / name)), [])
        with self.assertRaises(FileNotFoundError):
            self.model.predict(str(self.data / "missing.png"))

    def test_repeated_prediction_is_stable(self):
        path = str(self.data / "login.png")
        self.assertEqual(self.model.predict(path), self.model.predict(path))

    def test_corrupt_weights_report_error_without_masking_it(self):
        from eyeballer.model import EyeballModel
        output = io.StringIO()
        with tempfile.NamedTemporaryFile(suffix=".h5") as weights:
            weights.write(b"invalid weights")
            weights.flush()
            with redirect_stdout(output), self.assertRaises(SystemExit) as error:
                EyeballModel(weights_file=weights.name, quiet=True)
        self.assertNotEqual(error.exception.code, 0)
        self.assertIn("Unable to open weights file", output.getvalue())

    def test_missing_weights_fail_before_backbone_initialization(self):
        from eyeballer.model import EyeballModel
        import tensorflow as tf
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(tf.keras.applications.mobilenet, "MobileNet") as constructor:
                with self.assertRaises(FileNotFoundError):
                    EyeballModel(weights_file=str(Path(directory) / "missing.h5"))
                constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
