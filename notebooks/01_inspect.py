import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    # Marimo bootstrap; no rendered output.
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # CLIF Table Inspector

    Discovers CLIF-MIMIC parquet tables, shows inventory, and validates core CLIF contracts.
    """)
    return


@app.cell
def _(mo):
    mo.md(
        "**Step 1.** Load `configs/local.yaml` (falling back to the tracked example) "
        "and assert that `safety.allow_phi` is False."
    )
    return


@app.cell
def _(mo):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from _common import load_pipeline_config
    config, config_notice = load_pipeline_config(__file__, mo)
    config_notice
    return (config,)


@app.cell
def _(mo):
    mo.md(
        "**Step 2.** Discover every parquet under `config.data.root`, normalising names "
        "(stripping `clif_`/`clif2_`/etc.) so downstream code keys by bare table names."
    )
    return


@app.cell
def _(config, mo):
    from _common import discover_pipeline_tables
    tables = discover_pipeline_tables(config, mo)
    return (tables,)


@app.cell
def _(mo):
    mo.md("**Step 3.** Build a small inventory (table name, path, column count) for display.")
    return


@app.cell
def _(tables):
    from icumodelstream.io import table_inventory
    inventory = table_inventory(tables)
    return (inventory,)


@app.cell
def _(inventory, mo):
    mo.vstack([
        mo.md(f"**{len(inventory)} tables discovered**"),
        inventory,
    ])
    return


@app.cell
def _(mo):
    mo.md(
        "**Step 4.** Run the minimal CLIF contract validator against the core tables "
        "(`patient`, `hospitalization`, `adt`) so missing columns surface here, not deeper."
    )
    return


@app.cell
def _(tables):
    from icumodelstream.schema import validate_table_contracts, validation_results_to_frame
    results = validate_table_contracts(tables)
    validation_df = validation_results_to_frame(results)
    return (results, validation_df)


@app.cell
def _(mo, results, validation_df):
    n_ok = sum(r.ok for r in results)
    mo.vstack([
        mo.md(f"**{n_ok}/{len(results)} core tables passed validation**"),
        validation_df,
    ])
    return


if __name__ == "__main__":
    app.run()
