"""Paper-scoped PET/T1 prediction methods; no bundled participant data."""

from .modeling import paper_settings, run_nested_models
from .associations import association_family, bh_adjust, hc3_fit

__all__ = ["paper_settings", "run_nested_models", "association_family", "bh_adjust", "hc3_fit"]
__version__ = "0.2.0"
