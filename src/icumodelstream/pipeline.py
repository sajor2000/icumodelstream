"""End-to-end Phase 4 baseline pipeline.

Wraps cohort -> labels -> windowed features -> patient-aware split ->
LightGBM + logistic baselines into a single callable. The CLI in U2 wraps
this, and the marimo notebook (`notebooks/05_baseline.py`) is structurally
the same flow.

CLAUDE.md rule 2 (simplicity first): this module composes existing
building blocks. It does not introduce a new abstraction layer; it only
makes the "run the whole baseline" path importable so the CLI does not
have to re-implement orchestration.

CLAUDE.md rule 7 (fail loudly on data assumptions): missing required
tables (hospitalization, patient) propagate from io.scan_table; an empty
cohort, a degenerate label (all-zero / all-one), or no usable feature
tables raise ValueError before reaching the model layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import polars as pl

from icumodelstream.cohorts import (
    CohortSpec,
    CohortWaterfall,
    build_cohort_with_waterfall,
)
from icumodelstream.features import aggregate_numeric_table_windowed
from icumodelstream.io import TableRef, scan_table
from icumodelstream.labels import extract_mortality_labels
from icumodelstream.models import (
    BaselineResult,
    fit_lightgbm_baseline,
    fit_logistic_baseline,
)
from icumodelstream.splits import group_train_test_split


def get_admission_anchors(tables: dict[str, TableRef]) -> pl.DataFrame:
    """Return per-hospitalization admission_dttm renamed to anchor_dttm.

    Output schema: ``hospitalization_id``, ``anchor_dttm`` (Datetime). Ready
    to pass directly to :func:`aggregate_numeric_table_windowed`.

    Raises
    ------
    ValueError
        If the hospitalization table is missing the ``admission_dttm``
        column (CLAUDE.md rule 7).
    """
    lf = scan_table(tables, "hospitalization")
    columns = set(lf.collect_schema().names())
    if "admission_dttm" not in columns:
        raise ValueError(
            "hospitalization table is missing required column 'admission_dttm'; "
            f"observed columns: {sorted(columns)}."
        )
    return (
        lf.select("hospitalization_id", "admission_dttm")
        .rename({"admission_dttm": "anchor_dttm"})
        .collect()
    )


@dataclass(frozen=True)
class BaselinePipelineResult:
    """Everything a baseline run produces, in one bag.

    Fields mirror the marimo notebook's intermediate cells so the CLI can
    render the same summary without re-deriving values.
    """

    cohort_waterfall: CohortWaterfall
    n_features: int
    feature_names: list[str]
    n_train: int
    n_test: int
    n_train_patients: int
    n_test_patients: int
    train_prevalence: float
    test_prevalence: float
    lightgbm: BaselineResult
    lightgbm_model: Any
    logistic: BaselineResult
    config_snapshot: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def _try_windowed_features(
    tables: dict[str, TableRef],
    table_name: str,
    prefix: str,
    anchors: pl.DataFrame,
    window_hours: float,
    cohort: pl.DataFrame,
    warnings: list[str],
) -> pl.DataFrame | None:
    """Compute windowed features for a single table, or append a warning.

    Returns the feature DataFrame on success, ``None`` if the table is
    absent or the aggregation raised ValueError (in either case the reason
    is recorded in ``warnings`` so the caller surfaces it).
    """
    if table_name not in tables:
        warnings.append(f"{table_name} table not found in data root; skipping {prefix} features.")
        return None
    try:
        return aggregate_numeric_table_windowed(
            tables,
            table_name,
            prefix,
            anchors=anchors,
            window_hours=window_hours,
            cohort=cohort,
        )
    except ValueError as e:
        warnings.append(f"{table_name} features skipped: {type(e).__name__}: {e}")
        return None


def _build_feature_matrix(
    tables: dict[str, TableRef],
    cohort: pl.DataFrame,
    anchors: pl.DataFrame,
    window_hours: float,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Compute vitals + labs windowed features and join into a single matrix.

    Returns ``(matrix, feature_names, warnings)`` where ``matrix`` carries
    ``hospitalization_id`` plus all feature columns (no patient_id, no
    label yet) and ``feature_names`` lists feature columns in matrix order.

    Vitals is the LEFT base when present (typically the larger table), labs
    is left-joined on. If only one of the two is present, that one becomes
    the base. If neither is usable the caller raises -- this helper just
    reports an empty matrix.
    """
    warnings: list[str] = []
    vitals = _try_windowed_features(
        tables, "vitals", "vitals", anchors, window_hours, cohort, warnings
    )
    labs = _try_windowed_features(
        tables, "labs", "labs", anchors, window_hours, cohort, warnings
    )

    if vitals is not None and labs is not None:
        matrix = vitals.join(labs, on="hospitalization_id", how="left")
    elif vitals is not None:
        matrix = vitals
    elif labs is not None:
        matrix = labs
    else:
        return pl.DataFrame({"hospitalization_id": []}), [], warnings

    feature_names = [c for c in matrix.columns if c != "hospitalization_id"]
    return matrix, feature_names, warnings


def run_baseline_pipeline(
    tables: dict[str, TableRef],
    cohort_spec: CohortSpec,
    window_hours: float = 24.0,
    test_size: float = 0.2,
    seed: int = 42,
    include_hospice: bool = False,
) -> BaselinePipelineResult:
    """Run the full Phase 4 baseline: cohort -> labels -> features -> models.

    Steps mirror ``notebooks/05_baseline.py`` so the CLI and the notebook
    produce identical numbers from identical inputs.

    Parameters
    ----------
    tables:
        Discovered CLIF parquet tables (from :func:`discover_tables`).
    cohort_spec:
        Adult ICU cohort definition.
    window_hours:
        Feature aggregation window measured from admission (half-open).
    test_size, seed:
        Forwarded to :func:`group_train_test_split` for the patient-aware
        holdout.
    include_hospice:
        If True, hospitalizations discharged to hospice count as mortality=1.

    Raises
    ------
    ValueError
        Empty cohort, all-zero / all-one labels, or both vitals AND labs
        unusable (no features = no model).
    """
    config_snapshot: dict[str, Any] = {
        "cohort_spec": asdict(cohort_spec),
        "window_hours": window_hours,
        "test_size": test_size,
        "seed": seed,
        "include_hospice": include_hospice,
    }

    cohort, waterfall = build_cohort_with_waterfall(tables, cohort_spec)
    if cohort.height == 0:
        raise ValueError(
            "Cohort is empty after applying CohortSpec filters; cannot run baseline. "
            f"Waterfall: total={waterfall.total_hospitalizations}, "
            f"after_age={waterfall.after_age_filter}, "
            f"after_icu={waterfall.after_icu_filter}."
        )

    labels = extract_mortality_labels(tables, include_hospice=include_hospice)
    cohort_with_labels = cohort.join(labels, on="hospitalization_id", how="inner")
    if cohort_with_labels.height == 0:
        raise ValueError(
            "No cohort hospitalizations matched extracted labels; cannot run baseline."
        )

    anchors = get_admission_anchors(tables).join(
        cohort_with_labels.select("hospitalization_id"),
        on="hospitalization_id",
        how="inner",
    )

    matrix, feature_names, warnings = _build_feature_matrix(
        tables, cohort_with_labels, anchors, window_hours
    )
    if not feature_names:
        raise ValueError(
            "No usable feature tables (vitals and labs both missing or invalid); "
            f"cannot fit baseline. Warnings: {warnings}"
        )

    # Inner join on hospitalization_id so X / y / groups are perfectly aligned.
    full = matrix.join(
        cohort_with_labels.select("hospitalization_id", "patient_id", "mortality"),
        on="hospitalization_id",
        how="inner",
    ).sort("hospitalization_id")  # deterministic row order -> reproducible split

    X = full.select(feature_names)
    y = full["mortality"]
    groups = full["patient_id"]

    y_sum = int(y.sum())
    if y_sum == 0 or y_sum == y.len():
        raise ValueError(
            f"Label vector is degenerate (sum={y_sum}, n={y.len()}); cannot fit baseline. "
            "Need at least one example of each class."
        )

    # Attach patient_id as a non-feature column so we can recover unique
    # patient counts per side after splitting. We strip it off X_train /
    # X_test before handing them to the model layer.
    pid_col = "__patient_id"
    X_with_pid = X.with_columns(groups.alias(pid_col))
    X_with_pid_train, X_with_pid_test, y_train, y_test = group_train_test_split(
        X_with_pid, y, groups, test_size=test_size, seed=seed
    )
    n_train_patients = int(X_with_pid_train[pid_col].n_unique())
    n_test_patients = int(X_with_pid_test[pid_col].n_unique())
    X_train = X_with_pid_train.drop(pid_col)
    X_test = X_with_pid_test.drop(pid_col)

    n_train = X_train.height
    n_test = X_test.height
    train_prevalence = float(y_train.sum()) / n_train if n_train > 0 else 0.0
    test_prevalence = float(y_test.sum()) / n_test if n_test > 0 else 0.0

    lightgbm_model, lightgbm_result = fit_lightgbm_baseline(
        X_train, y_train, X_test, y_test, seed=seed
    )
    _, logistic_result = fit_logistic_baseline(
        X_train, y_train, X_test, y_test, seed=seed
    )

    return BaselinePipelineResult(
        cohort_waterfall=waterfall,
        n_features=len(feature_names),
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        n_train_patients=n_train_patients,
        n_test_patients=n_test_patients,
        train_prevalence=train_prevalence,
        test_prevalence=test_prevalence,
        lightgbm=lightgbm_result,
        lightgbm_model=lightgbm_model,
        logistic=logistic_result,
        config_snapshot=config_snapshot,
        warnings=warnings,
    )
