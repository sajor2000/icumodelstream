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
    # CLIF Table Inspector

    Discovers CLIF-MIMIC parquet tables, shows inventory, and validates core CLIF contracts.
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
