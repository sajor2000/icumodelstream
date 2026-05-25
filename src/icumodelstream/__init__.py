"""ICU Model Stream: local-first CLIF-MIMIC parquet pipeline."""

from icumodelstream._compat import patch_python_runtime_for_torch

patch_python_runtime_for_torch()

__all__ = ["__version__"]
__version__ = "0.1.0"
