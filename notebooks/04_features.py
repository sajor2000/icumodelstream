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
    # CLIF Baseline Features

    Per-hospitalization vitals and labs aggregates (mean, min, max, n) joined to the adult ICU cohort.
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
def _(features, mo):
    mo.vstack([
        mo.md(f"**Feature matrix: {features.height:,} rows × {features.width} columns**"),
        features.head(20),
    ])
    return


if __name__ == "__main__":
    app.run()
