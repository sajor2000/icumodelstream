"""Shared bootstrap helpers for the marimo notebooks.

Each notebook used to repeat ~25 lines of config-loading + safety-guard +
table-discovery boilerplate. Centralizing it here means a change to the
config search path or the safety contract requires editing one place.

These functions call `mo.stop(...)` on bootstrap failures, so the calling
cell halts gracefully with a readable message instead of crashing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from icumodelstream.config import AppConfig, load_config
from icumodelstream.io import TableRef, discover_tables

if TYPE_CHECKING:
    import marimo as mo  # noqa: F401  -- only for typing in helper signatures


def load_pipeline_config(notebook_file: str, mo: Any) -> tuple[AppConfig, Any]:
    """Load configs/local.yaml relative to the notebook, with example fallback.

    Returns ``(config, notice)``. ``notice`` is a marimo markdown element to display
    when the example fallback fired, or ``None`` when local.yaml was used. The caller
    should make ``notice`` (or a stack with it) the cell's last expression so the
    fallback is surfaced visibly per CLAUDE.md rule 7.

    Halts the cell via mo.stop on no-config-found, and on safety.allow_phi=True.
    """
    configs_dir = Path(notebook_file).parent.parent / "configs"
    config_path = configs_dir / "local.yaml"
    example_path = configs_dir / "local.example.yaml"
    notice: Any = None
    if config_path.exists():
        config = load_config(config_path)
    elif example_path.exists():
        config = load_config(example_path)
        notice = mo.md(
            f"ℹ️ `configs/local.yaml` not found — using `{example_path.name}`. "
            "Copy it to `local.yaml` and edit `data.root` for your machine."
        )
    else:
        mo.stop(
            True,
            mo.md(f"❌ No config file in `{configs_dir}`. Copy `local.example.yaml` to `local.yaml`."),
        )
    mo.stop(
        config.safety.allow_phi,
        mo.md("**Safety check failed:** `allow_phi` must be False"),
    )
    return config, notice


def discover_pipeline_tables(config: AppConfig, mo: Any) -> dict[str, TableRef]:
    """Discover CLIF parquet tables under config.data.root, halting on common failures.

    Converts FileNotFoundError (bad data root, empty directory) and ValueError
    (duplicate normalized keys) into mo.stop with a readable message.
    """
    try:
        return discover_tables(config.data.root, config.data.table_glob)
    except FileNotFoundError as e:
        mo.stop(True, mo.md(f"⚠️ Data root not found: `{config.data.root}`\n\n{e}"))
    except ValueError as e:
        mo.stop(True, mo.md(f"⚠️ Duplicate table names in data root:\n\n{e}"))
