"""Runtime compatibility shims for local development environments.

The project is intended for Python 3.11+ on Mac and Linux. The sandbox used for
repository validation reports Python 3.11 but lacks a small integer-string guard
API that recent PyTorch releases expect. Keep the shim tiny, explicit, and safe:
it only fills missing stdlib attributes with conservative no-op equivalents.
"""

from __future__ import annotations

import sys


def patch_python_runtime_for_torch() -> None:
    """Patch missing Python APIs expected by recent PyTorch builds.

    Python 3.11 release candidates may lack ``sys.get_int_max_str_digits`` and
    ``sys.set_int_max_str_digits``. PyTorch's Dynamo polyfills import these
    during optimizer construction, so provide no-op compatible functions when
    they are absent. Standard CPython 3.11+ installations are unaffected.
    """

    if not hasattr(sys, "get_int_max_str_digits"):
        sys.get_int_max_str_digits = lambda: 0  # type: ignore[attr-defined]
    if not hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits = lambda _maxdigits: None  # type: ignore[attr-defined]
