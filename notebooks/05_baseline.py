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
    # CLIF Phase 4: LightGBM Baseline for In-Hospital Mortality

    End-to-end pipeline: adult ICU cohort → in-hospital mortality label →
    first-24h vitals/labs aggregates → patient-level split → LightGBM + logistic
    baselines → AUROC/AUPRC/Brier + reliability diagram.

    **Success criterion (baseline-of-baseline):** LightGBM AUROC > 0.6 on the
    holdout. Calibration intercept ≈ 0 and slope ≈ 1 would mean well-calibrated
    probabilities; deviations are expected for an untuned baseline.

    **PHI:** This notebook displays summary statistics and a 10-bin calibration
    table only — no patient-level rows. Do NOT export to HTML/IPYNB.
    See CLAUDE.md > Data safety rules.
    """)
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
def _(config, mo):
    from _common import discover_pipeline_tables
    tables = discover_pipeline_tables(config, mo)
    return (tables,)


@app.cell
def _(config, mo, pl, tables):
    from icumodelstream.cohorts import CohortSpec, build_cohort_with_waterfall
    spec = CohortSpec(
        min_age=config.cohort.min_age,
        require_icu_location=config.cohort.require_icu_location,
    )
    try:
        cohort, waterfall = build_cohort_with_waterfall(tables, spec)
    except KeyError as e:
        mo.stop(True, mo.md(f"❌ Required CLIF table missing while building cohort: {e}"))

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
    # PHI note: aggregate counts only — no patient-level rows.
    mo.vstack([
        mo.md("## Cohort waterfall"),
        waterfall_df,
    ])
    return (cohort,)


@app.cell
def _(cohort, mo, pl, tables):
    from icumodelstream.labels import extract_mortality_labels
    try:
        labels = extract_mortality_labels(tables, include_hospice=False)
    except (KeyError, ValueError) as e:
        mo.stop(True, mo.md(f"❌ Could not extract mortality labels: {e}"))

    # Inner join keeps only cohort hospitalizations.
    cohort_with_labels = cohort.join(labels, on="hospitalization_id", how="inner")

    n_positive = int(cohort_with_labels["mortality"].sum())
    n_total = cohort_with_labels.height
    n_negative = n_total - n_positive
    prevalence = n_positive / n_total if n_total > 0 else 0.0
    balance_df = pl.DataFrame({
        "metric": ["n_positive", "n_negative", "prevalence"],
        "value": [float(n_positive), float(n_negative), float(prevalence)],
    })
    # PHI note: class balance counts only — aggregate, non-PHI.
    mo.vstack([
        mo.md("## Label class balance (in-hospital mortality)"),
        balance_df,
    ])
    return (cohort_with_labels,)


@app.cell
def _(cohort_with_labels, mo, pl, tables):
    from icumodelstream.io import scan_table
    try:
        hosp_lf = scan_table(tables, "hospitalization")
    except KeyError as e:
        mo.stop(True, mo.md(f"❌ hospitalization table missing for anchors: {e}"))

    hosp_cols = set(hosp_lf.collect_schema().names())
    if "admission_dttm" not in hosp_cols:
        mo.stop(
            True,
            mo.md(
                "❌ hospitalization table is missing `admission_dttm`; "
                "cannot build first-24h windows."
            ),
        )

    anchors = (
        hosp_lf.select("hospitalization_id", "admission_dttm")
        .collect()
        .rename({"admission_dttm": "anchor_dttm"})
        .join(
            cohort_with_labels.select("hospitalization_id"),
            on="hospitalization_id",
            how="inner",
        )
    )
    # PHI note: aggregate count only.
    mo.md(f"**Anchors built for {anchors.height:,} cohort hospitalizations.**")
    _ = pl  # keep pl alive for downstream cells
    return (anchors,)


@app.cell
def _(anchors, cohort_with_labels, mo, tables):
    from icumodelstream.features import aggregate_numeric_table_windowed
    vitals_features, vitals_warn = None, None
    if "vitals" in tables:
        try:
            vitals_features = aggregate_numeric_table_windowed(
                tables,
                "vitals",
                "vitals",
                anchors=anchors,
                window_hours=24,
                cohort=cohort_with_labels,
            )
        except (KeyError, ValueError) as e:
            vitals_warn = f"{type(e).__name__}: {e}"
    else:
        vitals_warn = "vitals table not found in data root"
    mo.md(f"⚠️ Vitals: {vitals_warn}") if vitals_warn else mo.md(
        f"✅ Vitals features computed ({vitals_features.width - 1} numeric columns)."
    )
    return (vitals_features,)


@app.cell
def _(anchors, cohort_with_labels, mo, tables):
    from icumodelstream.features import aggregate_numeric_table_windowed as _agg_windowed
    labs_features, labs_warn = None, None
    if "labs" in tables:
        try:
            labs_features = _agg_windowed(
                tables,
                "labs",
                "labs",
                anchors=anchors,
                window_hours=24,
                cohort=cohort_with_labels,
            )
        except (KeyError, ValueError) as e:
            labs_warn = f"{type(e).__name__}: {e}"
    else:
        labs_warn = "labs table not found in data root"
    mo.md(f"⚠️ Labs: {labs_warn}") if labs_warn else mo.md(
        f"✅ Labs features computed ({labs_features.width - 1} numeric columns)."
    )
    return (labs_features,)


@app.cell
def _(cohort_with_labels, labs_features, mo, vitals_features):
    if vitals_features is None and labs_features is None:
        mo.stop(
            True,
            mo.md("❌ Neither vitals nor labs features were available. Check data root."),
        )
    if vitals_features is not None and labs_features is not None:
        features = vitals_features.join(labs_features, on="hospitalization_id", how="left")
    elif vitals_features is not None:
        features = vitals_features
    else:
        features = labs_features

    # Join labels + patient_id onto features, then peel off X / y / groups.
    full = features.join(
        cohort_with_labels.select("hospitalization_id", "patient_id", "mortality"),
        on="hospitalization_id",
        how="inner",
    )
    feature_cols = [
        c for c in full.columns if c not in ("hospitalization_id", "patient_id", "mortality")
    ]
    X = full.select(feature_cols)
    y = full["mortality"]
    groups = full["patient_id"]
    # PHI note: shape / unique-patient counts are aggregate, non-PHI.
    mo.md(
        f"**Feature matrix shape:** {X.shape} | "
        f"**positives:** {int(y.sum()):,} | "
        f"**unique patients:** {groups.n_unique():,}"
    )
    return (X, groups, y)


@app.cell
def _(X, groups, mo, pl, y):
    from icumodelstream.splits import group_train_test_split

    # Use the same passenger-column pattern as pipeline.run_baseline_pipeline so
    # any change to the splitter (e.g., stratified-group) flows here automatically
    # without two parallel call sites drifting apart.
    pid_col = "__patient_id"
    if pid_col in X.columns:
        raise ValueError(f"Reserved sentinel column {pid_col!r} collides with a feature.")
    X_with_pid = X.with_columns(groups.alias(pid_col))
    X_with_pid_train, X_with_pid_test, y_train, y_test = group_train_test_split(
        X_with_pid, y, groups, test_size=0.2, seed=42
    )
    n_train_patients = int(X_with_pid_train[pid_col].n_unique())
    n_test_patients = int(X_with_pid_test[pid_col].n_unique())
    X_train = X_with_pid_train.drop(pid_col)
    X_test = X_with_pid_test.drop(pid_col)
    n_train = X_train.height
    n_test = X_test.height
    train_prevalence = float(y_train.sum()) / n_train if n_train else 0.0
    test_prevalence = float(y_test.sum()) / n_test if n_test else 0.0
    split_df = pl.DataFrame({
        "metric": [
            "n_train", "n_test", "n_train_patients", "n_test_patients",
            "train_prevalence", "test_prevalence",
        ],
        "value": [
            float(n_train), float(n_test),
            float(n_train_patients), float(n_test_patients),
            train_prevalence, test_prevalence,
        ],
    })
    # PHI note: split sizes / prevalences are aggregate, non-PHI.
    mo.vstack([
        mo.md("## Patient-aware split"),
        split_df,
    ])
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, X_train, y_test, y_train):
    from icumodelstream.models import fit_lightgbm_baseline
    lgbm_model, lgbm_result = fit_lightgbm_baseline(
        X_train, y_train, X_test, y_test, seed=42
    )
    return (lgbm_result,)


@app.cell
def _(X_test, X_train, y_test, y_train):
    from icumodelstream.models import fit_logistic_baseline
    logreg_model, logreg_result = fit_logistic_baseline(
        X_train, y_train, X_test, y_test, seed=42
    )
    return (logreg_result,)


@app.cell
def _(lgbm_result, logreg_result, mo, pl):
    metric_keys = (
        "auroc",
        "auprc",
        "brier_score",
        "prevalence",
        "calibration_intercept",
        "calibration_slope",
    )
    metrics_df = pl.DataFrame({
        "model": ["lightgbm", "logistic"],
        **{
            key: [
                float(lgbm_result.metrics[key]),
                float(logreg_result.metrics[key]),
            ]
            for key in metric_keys
        },
    })
    # PHI note: model-level metrics only — aggregate, non-PHI.
    mo.vstack([
        mo.md("## Holdout metrics"),
        metrics_df,
    ])
    return


@app.cell
def _(lgbm_result, logreg_result, mo):
    # PHI note: calibration tables aggregate test-set predictions into 10 bins —
    # no patient-level rows are displayed.
    mo.hstack([
        mo.vstack([mo.md("## LightGBM reliability"), lgbm_result.calibration_table]),
        mo.vstack([mo.md("## Logistic reliability"), logreg_result.calibration_table]),
    ])
    return


@app.cell
def _(lgbm_result, logreg_result, mo):
    lgbm_auroc = lgbm_result.metrics["auroc"]
    logreg_auroc = logreg_result.metrics["auroc"]
    better = "LightGBM" if lgbm_auroc >= logreg_auroc else "logistic regression"

    def _calibration_verdict(intercept: float, slope: float) -> str:
        # Rough heuristic per the U6 plan: |intercept| < 0.5 and |slope - 1| < 0.5.
        if abs(intercept) < 0.5 and abs(slope - 1.0) < 0.5:
            return "reasonably calibrated"
        return "miscalibrated (expected for an untuned baseline)"

    lgbm_int = lgbm_result.metrics["calibration_intercept"]
    lgbm_slope = lgbm_result.metrics["calibration_slope"]
    logreg_int = logreg_result.metrics["calibration_intercept"]
    logreg_slope = logreg_result.metrics["calibration_slope"]
    lgbm_cal = _calibration_verdict(lgbm_int, lgbm_slope)
    logreg_cal = _calibration_verdict(logreg_int, logreg_slope)

    mo.md(
        f"""
        ## Summary

        - **Higher AUROC:** {better} (LightGBM={lgbm_auroc:.3f}, logistic={logreg_auroc:.3f}).
        - **LightGBM calibration:** {lgbm_cal}
          (intercept={lgbm_int:.2f}, slope={lgbm_slope:.2f}).
        - **Logistic calibration:** {logreg_cal}
          (intercept={logreg_int:.2f}, slope={logreg_slope:.2f}).

        **Next steps:** tune hyperparameters with patient-aware K-fold CV (see
        `icumodelstream.splits.group_kfold`), add medications / respiratory-support
        features, then expand to time-series modeling. Per CLAUDE.md rule 9,
        deep-learning work waits until these baselines are reproducible and
        clinically reviewed.
        """
    )
    return


if __name__ == "__main__":
    app.run()
