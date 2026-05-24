import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # CLIF QC Report

    Row counts and missingness for all discovered CLIF-MIMIC tables.
    """)
    return


@app.cell
def _(mo):
    from pathlib import Path
    from icumodelstream.config import load_config
    configs_dir = Path(__file__).parent.parent / "configs"
    config_path = configs_dir / "local.yaml"
    example_path = configs_dir / "local.example.yaml"
    if config_path.exists():
        config = load_config(config_path)
        config_notice = None
    elif example_path.exists():
        config = load_config(example_path)
        config_notice = mo.md(
            f"ℹ️ `configs/local.yaml` not found — using `{example_path.name}`. "
            "Copy it to `local.yaml` and edit `data.root` for your machine."
        )
    else:
        mo.stop(True, mo.md(f"❌ No config file in `{configs_dir}`. Copy `local.example.yaml` to `local.yaml`."))
    mo.stop(config.safety.allow_phi, mo.md("**Safety check failed:** `allow_phi` must be False"))
    config_notice
    return (config,)


@app.cell
def _(config, mo):
    from icumodelstream.io import discover_tables
    try:
        tables = discover_tables(config.data.root, config.data.table_glob)
    except FileNotFoundError as e:
        mo.stop(True, mo.md(f"⚠️ Data root not found: `{config.data.root}`\n\n{e}"))
    except ValueError as e:
        mo.stop(True, mo.md(f"⚠️ Duplicate table names in data root:\n\n{e}"))
    return (tables,)


@app.cell
def _(tables):
    from icumodelstream.qc import build_qc_report
    report = build_qc_report(tables)
    return (report,)


@app.cell
def _(mo, report):
    import polars as pl
    sections = []
    for tbl in report["tables"]:
        n_rows = tbl["n_rows"]
        miss = tbl["missing_first_columns"]
        n_checked = len(miss)
        n_total = tbl["n_columns"]
        header = f"## {tbl['table']} — {n_rows:,} rows × {n_total} columns"
        if n_checked < n_total:
            header += f" (QC inspected first {n_checked})"
        nonzero = {k: v for k, v in miss.items() if v > 0} if n_rows > 0 else {}
        if nonzero:
            miss_df = pl.DataFrame({
                "column": list(nonzero.keys()),
                "n_null": list(nonzero.values()),
                "n_rows": [n_rows] * len(nonzero),
                "pct_null": [round(v / n_rows * 100, 1) for v in nonzero.values()],
            })
            body = miss_df
        else:
            body = mo.md(
                "_No null values in inspected columns._"
                if n_rows > 0
                else "_Empty table — 0 rows._"
            )
        sections.append(mo.vstack([mo.md(header), body]))
    mo.vstack(sections)
    return


if __name__ == "__main__":
    app.run()
