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
    # CLIF Baseline Features

    Per-hospitalization vitals and labs aggregates (mean, min, max, n) joined to the adult ICU cohort.
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
        "**Step 3.** Build the adult ICU cohort. We use the original `build_adult_icu_cohort` "
        "here (not the waterfall variant) because features only need the final cohort frame."
    )
    return


@app.cell
def _(config, mo, tables):
    from icumodelstream.cohorts import CohortSpec, build_adult_icu_cohort
    try:
        cohort = build_adult_icu_cohort(
            tables,
            CohortSpec(
                min_age=config.cohort.min_age,
                require_icu_location=config.cohort.require_icu_location,
            ),
        )
    except KeyError as e:
        mo.stop(True, mo.md(f"❌ Required CLIF table missing while building cohort: {e}"))
    return (cohort,)


@app.cell
def _(mo):
    mo.md(
        "**Step 4a.** Aggregate vitals per hospitalization into mean/min/max/n. Anchored "
        "to the cohort: hospitalizations with no vitals get null aggregates rather than being dropped."
    )
    return


@app.cell
def _(cohort, mo, tables):
    from icumodelstream.features import aggregate_numeric_table as _aggregate_numeric_table
    vitals_features, vitals_warn = None, None
    if "vitals" in tables:
        try:
            vitals_features = _aggregate_numeric_table(tables, "vitals", "vitals", cohort=cohort)
        except Exception as e:
            vitals_warn = f"{type(e).__name__}: {e}"
    else:
        vitals_warn = "vitals table not found in data root"
    mo.md(f"⚠️ Vitals: {vitals_warn}") if vitals_warn else mo.md("✅ Vitals features computed.")
    return (vitals_features, vitals_warn)


@app.cell
def _(mo):
    mo.md("**Step 4b.** Same aggregation for labs.")
    return


@app.cell
def _(cohort, mo, tables):
    from icumodelstream.features import aggregate_numeric_table as _aggregate_numeric_table
    labs_features, labs_warn = None, None
    if "labs" in tables:
        try:
            labs_features = _aggregate_numeric_table(tables, "labs", "labs", cohort=cohort)
        except Exception as e:
            labs_warn = f"{type(e).__name__}: {e}"
    else:
        labs_warn = "labs table not found in data root"
    mo.md(f"⚠️ Labs: {labs_warn}") if labs_warn else mo.md("✅ Labs features computed.")
    return (labs_features, labs_warn)


@app.cell
def _(mo):
    mo.md(
        "**Step 5.** Combine vitals and labs into a single feature matrix. Both sides "
        "are already cohort-anchored so a left join preserves all cohort rows."
    )
    return


@app.cell
def _(labs_features, mo, vitals_features):
    if vitals_features is None and labs_features is None:
        mo.stop(True, mo.md("❌ Neither vitals nor labs features were available. Check data root."))
    if vitals_features is not None and labs_features is not None:
        features = vitals_features.join(labs_features, on="hospitalization_id", how="left")
    elif vitals_features is not None:
        features = vitals_features
    else:
        features = labs_features
    return (features,)


@app.cell
def _(mo):
    mo.md(
        "**Step 6.** Preview the first 20 rows of the feature matrix. "
        "**PHI note:** this preview renders only in the live marimo session — `.py` notebooks "
        "store no outputs. Do NOT run `marimo export html/ipynb` on this notebook against "
        "real CLIF-MIMIC data; the resulting file would embed hospitalization_id and per-stay "
        "aggregates. See CLAUDE.md > Data safety rules."
    )
    return


@app.cell
def _(features, mo):
    mo.vstack([
        mo.md(f"**Feature matrix: {features.height:,} rows × {features.width} columns**"),
        features.head(20),
    ])
    return


if __name__ == "__main__":
    app.run()
