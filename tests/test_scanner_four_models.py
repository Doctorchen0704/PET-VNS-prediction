"""Expose the scanner-extension synthetic tests to standard test discovery."""
import importlib.util
from pathlib import Path
import sys
import unittest

EXTENSION = Path(__file__).resolve().parents[1] / "STEP07_prediction_models" / "scanner_adjusted_four_models"
sys.path.insert(0, str(EXTENSION))


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name in ("test_pipeline", "test_scanner_solver"):
        spec = importlib.util.spec_from_file_location("scanner_public_" + name, EXTENSION / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        suite.addTests(loader.loadTestsFromModule(module))
    return suite
