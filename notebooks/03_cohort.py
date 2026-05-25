import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    # Marimo bootstrap; no rendered output.
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
    mo.md("**Step 2.** Discover the CLIF parquet tables under `config.data.root`.")
    return


@app.cell
def _(config, mo):
    from _common import discover_pipeline_tables
    tables = discover_pipeline_tables(config, mo)
    return (tables,)


@app.cell
def _(mo):
    mo.md(
        "**Step 3.** Build the adult ICU cohort via `build_cohort_with_waterfall`. The "
        "waterfall records the row count surviving each filter step (total → age → ICU → "
        "final unique hospitalizations) plus which age/ICU columns were resolved."
    )
    return


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
def _(mo):
    mo.md(
        "**Step 4.** Derive the human-readable step labels. Labels reflect the actual "
        "filter outcome (e.g. 'Age filter SKIPPED — no age column found' when AGE_CANDIDATES "
        "didn't resolve) so the operator can tell skipped filters from applied ones."
    )
    return


@app.cell
def _(config, waterfall):
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
    return (age_step, icu_step)


@app.cell
def _(mo):
    mo.md("**Step 5.** Render the waterfall as a 4-row table.")
    return


@app.cell
def _(age_step, icu_step, mo, pl, waterfall):
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
def _(mo):
    mo.md(
        "**Step 6.** Preview the final cohort (first 20 rows). "
        "**PHI note:** this preview renders only in the live marimo session — `.py` notebooks "
        "store no outputs. Do NOT run `marimo export html/ipynb` on this notebook against "
        "real CLIF-MIMIC data; the resulting file would embed patient_id + hospitalization_id + "
        "age. See CLAUDE.md > Data safety rules."
    )
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
