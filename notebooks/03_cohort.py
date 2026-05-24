import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return (mo, pl)


@app.cell
def _(mo):
    mo.md(r"""
    # CLIF Adult ICU Cohort

    Builds the adult ICU cohort and shows a row-count waterfall for each filter step.
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
def _(config, mo, tables):
    from icumodelstream.cohorts import CohortSpec, build_cohort_with_waterfall
    spec = CohortSpec(
        min_age=config.cohort.min_age,
        require_icu_location=config.cohort.require_icu_location,
    )
    try:
        cohort, waterfall = build_cohort_with_waterfall(tables, spec)
    except KeyError as e:
        mo.stop(True, mo.md(f"❌ Required CLIF table missing: {e}"))
    return (cohort, spec, waterfall)


@app.cell
def _(config, mo, pl, waterfall):
    age_step = (
        f"After age ≥ {config.cohort.min_age}"
        if waterfall.age_col_used is not None
        else "Age filter SKIPPED — no age column found"
    )
    if waterfall.icu_filter_applied:
        icu_step = f"After ICU location filter ({waterfall.icu_location_col_used})"
    elif not config.cohort.require_icu_location:
        icu_step = "ICU filter disabled in config"
    else:
        icu_step = "ICU filter SKIPPED — no ADT or location column found"
    waterfall_df = pl.DataFrame({
        "step": ["All hospitalizations", age_step, icu_step, "Final cohort"],
        "n": [
            waterfall.total_hospitalizations,
            waterfall.after_age_filter,
            waterfall.after_icu_filter,
            waterfall.final,
        ],
    })
    mo.vstack([
        mo.md("## Cohort waterfall"),
        waterfall_df,
    ])
    return


@app.cell
def _(cohort, mo):
    mo.vstack([
        mo.md(f"**Final cohort: {cohort.height:,} hospitalizations**"),
        cohort.head(20),
    ])
    return


if __name__ == "__main__":
    app.run()
