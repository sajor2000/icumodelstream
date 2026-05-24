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
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from _common import load_pipeline_config
    config = load_pipeline_config(__file__, mo)
    return (config,)


@app.cell
def _(config, mo):
    from _common import discover_pipeline_tables
    tables = discover_pipeline_tables(config, mo)
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
