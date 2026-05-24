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
    config = load_config(Path(__file__).parent.parent / "configs" / "local.yaml")
    mo.stop(config.safety.allow_phi, mo.md("**Safety check failed:** `allow_phi` must be False"))
    return (config,)


@app.cell
def _(config, mo):
    from icumodelstream.io import discover_tables
    try:
        tables = discover_tables(config.data.root, config.data.table_glob)
    except FileNotFoundError as e:
        mo.stop(True, mo.md(f"⚠️ Data root not found: `{config.data.root}`\n\n{e}"))
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
        if n_rows > 0 and miss:
            miss_df = pl.DataFrame({
                "column": list(miss.keys()),
                "n_null": list(miss.values()),
                "n_rows": [n_rows] * len(miss),
                "pct_null": [round(v / n_rows * 100, 1) for v in miss.values()],
            })
        else:
            miss_df = pl.DataFrame({"column": [], "n_null": [], "n_rows": [], "pct_null": []})
        sections.append(mo.vstack([
            mo.md(f"## {tbl['table']} — {n_rows:,} rows × {tbl['n_columns']} columns"),
            miss_df,
        ]))
    mo.vstack(sections)
    return


if __name__ == "__main__":
    app.run()
