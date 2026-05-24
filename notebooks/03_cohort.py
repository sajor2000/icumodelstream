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
def _(config, mo, pl, tables):
    from icumodelstream.io import scan_table
    from icumodelstream.cohorts import AGE_CANDIDATES, first_existing_column

    try:
        hosp_lf = scan_table(tables, "hospitalization")
        patient_lf = scan_table(tables, "patient")
    except KeyError as e:
        mo.stop(True, mo.md(f"❌ Required CLIF table missing: {e}"))

    total_hosp = hosp_lf.select(pl.col("hospitalization_id").n_unique()).collect().item()

    patient_cols = set(patient_lf.collect_schema().names())
    hosp_cols = set(hosp_lf.collect_schema().names())
    age_col = first_existing_column(patient_cols | hosp_cols, AGE_CANDIDATES)

    if age_col is not None:
        after_age = (
            hosp_lf.join(patient_lf, on="patient_id", how="left")
            .filter(pl.col(age_col) >= config.cohort.min_age)
            .select(pl.col("hospitalization_id").n_unique())
            .collect()
            .item()
        )
    else:
        after_age = total_hosp

    return (after_age, age_col, total_hosp)


@app.cell
def _(config, tables):
    from icumodelstream.cohorts import CohortSpec, build_adult_icu_cohort
    spec = CohortSpec(
        min_age=config.cohort.min_age,
        require_icu_location=config.cohort.require_icu_location,
    )
    cohort = build_adult_icu_cohort(tables, spec)
    return (cohort, spec)


@app.cell
def _(after_age, age_col, cohort, config, mo, pl, total_hosp):
    age_step = (
        f"After age ≥ {config.cohort.min_age}"
        if age_col is not None
        else "Age filter SKIPPED — no age column found"
    )
    icu_step = (
        "After ICU location filter"
        if config.cohort.require_icu_location
        else "ICU filter disabled in config"
    )
    waterfall = pl.DataFrame({
        "step": ["All hospitalizations", age_step, icu_step],
        "n": [total_hosp, after_age, cohort.height],
    })
    mo.vstack([
        mo.md("## Cohort waterfall"),
        waterfall,
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
