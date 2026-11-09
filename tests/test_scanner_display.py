"""Expose the scanner-adjusted display tests to ordinary unittest discovery."""
import importlib.util
from pathlib import Path
import sys


def load_tests(loader, tests, pattern):
    path = (Path(__file__).resolve().parents[1] / 'STEP10_tables_and_figures' /
            'scanner_four_model_tests' / 'test_display.py')
    spec = importlib.util.spec_from_file_location('scanner_four_model_display_tests', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return loader.loadTestsFromModule(module)
