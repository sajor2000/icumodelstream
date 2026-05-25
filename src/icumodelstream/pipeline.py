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
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    # Type-only import: at runtime ``pipeline`` must not import ``sequences``
    # / ``torch_models`` / ``torch_train`` because ``sequences`` already
    # imports the RICH_* constants from this module (cycle). The function
    # body of ``run_sequence_baseline_pipeline`` imports them lazily.
    from icumodelstream.torch_models import SequenceResult

from icumodelstream.cohorts import (
    CohortSpec,
    CohortWaterfall,
    build_cohort_with_waterfall,
)
from icumodelstream.features import (
    aggregate_numeric_table_per_category,
    aggregate_numeric_table_windowed,
    respiratory_support_indicator,
)
from icumodelstream.io import TableRef, scan_table
from icumodelstream.labels import extract_los_label, extract_mortality_labels
from icumodelstream.models import (
    BaselineResult,
    fit_lightgbm_baseline,
    fit_logistic_baseline,
)
from icumodelstream.splits import group_train_test_split

# Rich feature-set vocabularies, validated against the observed CLIF-MIMIC
# data dictionary (docs/data_dictionary_notes.md). Each tuple lists the
# CLIF category values that are common enough across MIMIC to be worth
# including. Adding a category outside the observed vocabulary is harmless
# (the aggregation simply produces _n=0 with null mean/min/max) so this
# list can grow without breaking the pipeline.

RICH_VITAL_CATEGORIES: tuple[str, ...] = (
    "heart_rate", "sbp", "dbp", "map", "respiratory_rate", "spo2", "temp_c",
)
RICH_LAB_CATEGORIES: tuple[str, ...] = (
    "sodium", "potassium", "chloride", "bicarbonate", "bun", "creatinine",
    "glucose_serum", "calcium_total", "magnesium", "lactate", "hemoglobin", "wbc",
)
RICH_ASSESSMENT_CATEGORIES: tuple[str, ...] = ("gcs_total", "RASS")
RICH_RESPIRATORY_DEVICES: tuple[str, ...] = (
    "IMV", "NIPPV", "High Flow NC", "Nasal Cannula", "CPAP",
)


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
        .unique(subset=["hospitalization_id"])
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
    # Hospitalization IDs of the test fold, in the same order as
    # ``lightgbm.y_true`` / ``lightgbm.y_pred_proba``. Required for subgroup
    # analysis, decision-curve analysis, and external validation -- all
    # need to map per-row predictions back to per-row CLIF identity.
    test_hospitalization_ids: list[str] = field(default_factory=list)
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
    """Compute windowed features for a single table, or warn and return None.

    "Missing table" (a benign 'this site didn't export labs') is downgraded to a warning
    so the pipeline can run with whichever tables ARE present. Any other ValueError
    (parse failure, dtype mismatch, missing value column) is re-raised — per CLAUDE.md
    rule 7 these are data-quality red flags the operator must see, not silent skips.
    """
    if table_name not in tables:
        warnings.append(f"{table_name} table not found in data root; skipping {prefix} features.")
        return None
    return aggregate_numeric_table_windowed(
        tables,
        table_name,
        prefix,
        anchors=anchors,
        window_hours=window_hours,
        cohort=cohort,
    )


def _build_feature_matrix(
    tables: dict[str, TableRef],
    cohort: pl.DataFrame,
    anchors: pl.DataFrame,
    window_hours: float,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Basic feature set: vitals + labs aggregated as single columns each.

    Returns ``(matrix, feature_names, warnings)``. Produces 8 features when
    both vitals and labs are present (mean/min/max/n for each).
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


def _try_per_category(
    tables: dict[str, TableRef],
    table_name: str,
    category_column: str,
    categories: tuple[str, ...],
    prefix_template: str,
    anchors: pl.DataFrame,
    window_hours: float,
    cohort: pl.DataFrame,
    warnings: list[str],
) -> pl.DataFrame | None:
    """Per-category windowed aggregation with graceful skip on missing table."""
    if table_name not in tables:
        warnings.append(
            f"{table_name} table not found; skipping per-category features."
        )
        return None
    try:
        return aggregate_numeric_table_per_category(
            tables,
            table_name,
            category_column=category_column,
            categories=list(categories),
            prefix_template=prefix_template,
            anchors=anchors,
            window_hours=window_hours,
            cohort=cohort,
        )
    except ValueError as e:
        warnings.append(
            f"{table_name} per-category features skipped: {type(e).__name__}: {e}"
        )
        return None


def _build_rich_feature_matrix(
    tables: dict[str, TableRef],
    cohort: pl.DataFrame,
    anchors: pl.DataFrame,
    window_hours: float,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Rich feature set: per-category vitals + labs + assessments + resp flags.

    With the default RICH_* vocabularies this produces roughly 7+12+2 = 21
    "channels" * 4 stats = 84 numeric features plus 5 respiratory device
    indicator flags, for ~89 features total. Missing tables produce
    warnings rather than failure (the model layer still gets all categories
    in the schema, just with _n=0 and null aggregates for absent ones).
    """
    warnings: list[str] = []
    parts: list[pl.DataFrame] = []

    vitals = _try_per_category(
        tables, "vitals", "vital_category", RICH_VITAL_CATEGORIES,
        "vitals_{category}", anchors, window_hours, cohort, warnings,
    )
    if vitals is not None:
        parts.append(vitals)

    labs = _try_per_category(
        tables, "labs", "lab_category", RICH_LAB_CATEGORIES,
        "labs_{category}", anchors, window_hours, cohort, warnings,
    )
    if labs is not None:
        parts.append(labs)

    assessments = _try_per_category(
        tables, "patient_assessments", "assessment_category",
        RICH_ASSESSMENT_CATEGORIES, "assess_{category}",
        anchors, window_hours, cohort, warnings,
    )
    if assessments is not None:
        parts.append(assessments)

    if "respiratory_support" in tables:
        try:
            resp = respiratory_support_indicator(
                tables,
                device_categories=list(RICH_RESPIRATORY_DEVICES),
                anchors=anchors,
                window_hours=window_hours,
                cohort=cohort,
            )
            parts.append(resp)
        except ValueError as e:
            warnings.append(
                f"respiratory_support indicators skipped: {type(e).__name__}: {e}"
            )
    else:
        warnings.append(
            "respiratory_support table not found; skipping IMV/NIPPV/CPAP flags."
        )

    if not parts:
        return pl.DataFrame({"hospitalization_id": []}), [], warnings

    matrix = parts[0]
    for part in parts[1:]:
        matrix = matrix.join(part, on="hospitalization_id", how="left")

    feature_names = [c for c in matrix.columns if c != "hospitalization_id"]
    return matrix, feature_names, warnings


def run_baseline_pipeline(
    tables: dict[str, TableRef],
    cohort_spec: CohortSpec,
    window_hours: float = 24.0,
    test_size: float = 0.2,
    seed: int = 42,
    include_hospice: bool = False,
    feature_set: str = "basic",
    outcome: str = "mortality",
    los_threshold_hours: float = 168.0,
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
    feature_set:
        Either ``"basic"`` (single mean/min/max/n for vitals + labs; 8 features)
        or ``"rich"`` (per-category vitals + labs + GCS/RASS assessments + IMV/
        NIPPV/CPAP/etc. respiratory flags; ~85 features). Rich is recommended
        for actual modeling; basic is preserved for backwards compatibility
        with prior runs and the original Phase 4 plan numbers.
    outcome:
        Which label to predict: ``"mortality"`` (in-hospital death, default) or
        ``"los_gt_7d"`` (prolonged length of stay, threshold parameterized via
        ``los_threshold_hours``). Both are extracted from the hospitalization
        table; LOS uses ``discharge_dttm - admission_dttm``.
    los_threshold_hours:
        Threshold in hours for the LOS outcome. Default 168.0 (= 7 days).
        Ignored when ``outcome == "mortality"``.

    Raises
    ------
    ValueError
        Empty cohort, all-zero / all-one labels, no usable feature tables,
        or an unrecognized ``feature_set``.
    """
    if feature_set not in {"basic", "rich"}:
        raise ValueError(
            f"feature_set must be 'basic' or 'rich', got {feature_set!r}."
        )
    if outcome not in {"mortality", "los_gt_7d"}:
        raise ValueError(
            f"outcome must be 'mortality' or 'los_gt_7d', got {outcome!r}."
        )

    config_snapshot: dict[str, Any] = {
        "cohort_spec": asdict(cohort_spec),
        "window_hours": window_hours,
        "test_size": test_size,
        "seed": seed,
        "include_hospice": include_hospice,
        "feature_set": feature_set,
        "outcome": outcome,
        "los_threshold_hours": los_threshold_hours,
    }

    cohort, waterfall = build_cohort_with_waterfall(tables, cohort_spec)
    if cohort.height == 0:
        raise ValueError(
            "Cohort is empty after applying CohortSpec filters; cannot run baseline. "
            f"Waterfall: total={waterfall.total_hospitalizations}, "
            f"after_age={waterfall.after_age_filter}, "
            f"after_icu={waterfall.after_icu_filter}."
        )

    if outcome == "mortality":
        raw_labels = extract_mortality_labels(tables, include_hospice=include_hospice)
        # Rename to a consistent internal column so downstream code never branches on outcome name.
        labels = raw_labels.rename({"mortality": "outcome"})
    else:  # outcome == "los_gt_7d"
        raw_labels = extract_los_label(tables, threshold_hours=los_threshold_hours)
        labels = raw_labels.rename({"long_los": "outcome"})
    cohort_id_dtype = cohort.schema["hospitalization_id"]
    label_id_dtype = labels.schema["hospitalization_id"]
    if cohort_id_dtype != label_id_dtype:
        raise ValueError(
            f"hospitalization_id dtype mismatch between cohort ({cohort_id_dtype}) and "
            f"labels ({label_id_dtype}); cast both sides to a common type before joining."
        )
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

    if feature_set == "rich":
        matrix, feature_names, warnings = _build_rich_feature_matrix(
            tables, cohort_with_labels, anchors, window_hours
        )
    else:
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
        cohort_with_labels.select("hospitalization_id", "patient_id", "outcome"),
        on="hospitalization_id",
        how="inner",
    ).sort("hospitalization_id")  # deterministic row order -> reproducible split

    X = full.select(feature_names)
    y = full["outcome"]
    groups = full["patient_id"]

    y_sum = int(y.sum())
    if y_sum == 0 or y_sum == y.len():
        raise ValueError(
            f"Label vector is degenerate (sum={y_sum}, n={y.len()}); cannot fit baseline. "
            "Need at least one example of each class."
        )

    # Attach patient_id + hospitalization_id as non-feature sentinel columns
    # so we can recover unique patient counts per side AND the per-row test
    # identity (subgroup analysis, decision-curve analysis, external
    # validation all need to map per-row predictions back to identity).
    # Strip both sentinels before handing X to the model layer.
    pid_col = "__patient_id"
    hid_col = "__hospitalization_id"
    for sentinel in (pid_col, hid_col):
        if sentinel in X.columns:
            raise ValueError(
                f"Reserved sentinel column {sentinel!r} collides with an existing feature; "
                "rename the upstream feature or report this as a bug in run_baseline_pipeline."
            )
    X_with_ids = X.with_columns(
        groups.alias(pid_col),
        full["hospitalization_id"].alias(hid_col),
    )
    X_with_ids_train, X_with_ids_test, y_train, y_test = group_train_test_split(
        X_with_ids, y, groups, test_size=test_size, seed=seed
    )
    n_train_patients = int(X_with_ids_train[pid_col].n_unique())
    n_test_patients = int(X_with_ids_test[pid_col].n_unique())
    test_hospitalization_ids = X_with_ids_test[hid_col].to_list()
    X_train = X_with_ids_train.drop([pid_col, hid_col])
    X_test = X_with_ids_test.drop([pid_col, hid_col])

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
        test_hospitalization_ids=test_hospitalization_ids,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Phase 5 / U4 Sprint 4a: sequence-baseline orchestrator.
#
# Same cohort + labels path as ``run_baseline_pipeline``, but builds a
# (n_hosps, window_hours, n_channels) tensor via ``build_sequence_tensors``
# and trains the LSTM via ``fit_sequence_model``. The result type mirrors
# ``BaselinePipelineResult`` field-for-field where possible so downstream
# JSON / Markdown writers can dispatch on result type without rewriting.
#
# CLAUDE.md rule 2 (simplicity first): this is composition of already-tested
# pieces (build_cohort_with_waterfall -> labels -> build_sequence_tensors ->
# fit_sequence_model). No new abstraction layer; no model code lives here.
# CLAUDE.md rule 7 (fail loudly): outcome dispatch + degenerate-label /
# empty-cohort guards mirror ``run_baseline_pipeline`` exactly.
# CLAUDE.md rule 8 (no silent patient leakage): the 70/15/15 split happens
# inside ``prepare_split_tensors`` (called by both our prevalence-recovery
# step and ``fit_sequence_model``) using ``patient_id`` as the group key.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequencePipelineResult:
    """End-to-end Phase 5 sequence-baseline result.

    Fields mirror :class:`BaselinePipelineResult` where possible so the CLI's
    JSON and markdown writers can branch on result type without rewriting.
    The sequence model uses a 70/15/15 train/val/test split (vs the flat
    baseline's 80/20 train/test), so the dataclass exposes three counts
    instead of two.
    """

    cohort_waterfall: CohortWaterfall
    n_channels: int
    channel_names: list[str]
    n_train: int
    n_val: int
    n_test: int
    n_train_patients: int
    n_val_patients: int
    n_test_patients: int
    train_prevalence: float
    val_prevalence: float
    test_prevalence: float
    lstm: SequenceResult
    lstm_model: Any
    config_snapshot: dict[str, Any]
    # Hospitalization IDs of the test fold, in the same order as
    # ``lstm.y_true`` / ``lstm.y_pred_proba``. Recovered from the
    # ``SplitTensors.test_indices`` returned by ``prepare_split_tensors``
    # indexed back into the source ``sequences.hospitalization_ids``.
    test_hospitalization_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def run_sequence_baseline_pipeline(
    tables: dict[str, TableRef],
    cohort_spec: CohortSpec,
    *,
    window_hours: int = 24,
    seed: int = 42,
    outcome: str = "mortality",
    include_hospice: bool = False,
    los_threshold_hours: float = 168.0,
    hidden_dim: int = 128,
    n_layers: int = 2,
    dropout: float = 0.3,
    max_epochs: int = 20,
    patience: int = 3,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    device: str | None = None,
) -> SequencePipelineResult:
    """Run the full Phase 5 sequence baseline: cohort -> labels -> tensors -> LSTM.

    Mirrors :func:`run_baseline_pipeline`'s cohort + label flow but builds an
    ``(n_hosps, window_hours, n_channels)`` tensor instead of a flat feature
    matrix and trains an LSTM via :func:`fit_sequence_model`. Always uses the
    rich channel set (``RICH_VITAL_CATEGORIES`` + ``RICH_LAB_CATEGORIES`` +
    ``RICH_ASSESSMENT_CATEGORIES`` + ``RICH_RESPIRATORY_DEVICES``); there is
    no ``feature_set`` knob because the sequence model is the rich-features
    counterpart by construction.

    Parameters
    ----------
    tables:
        Discovered CLIF parquet tables (from :func:`discover_tables`).
    cohort_spec:
        Adult ICU cohort definition.
    window_hours:
        Length of the per-hospitalization window in integer hours. Forwarded
        to :func:`build_sequence_tensors`.
    seed:
        Forwarded to :func:`fit_sequence_model` (and internally to
        :func:`prepare_split_tensors`). Same seed -> reproducible split on CPU.
    outcome:
        Either ``"mortality"`` (in-hospital death, default) or ``"los_gt_7d"``
        (prolonged length-of-stay; threshold from ``los_threshold_hours``).
    include_hospice:
        If True, hospitalizations discharged to hospice count as mortality=1.
        Ignored when ``outcome != "mortality"``.
    los_threshold_hours:
        Threshold in hours for the LOS outcome. Default 168.0 (= 7 days).
        Ignored when ``outcome == "mortality"``.
    hidden_dim, n_layers, dropout:
        :class:`LSTMBaseline` architecture hyperparameters.
    max_epochs, patience, learning_rate, weight_decay, batch_size:
        :func:`train_lstm` hyperparameters.
    device:
        ``"cpu"``, ``"mps"``, ``"cuda"``, or ``None``. ``None`` lets
        :func:`fit_sequence_model` autodetect the best available device.

    Raises
    ------
    ValueError
        Unknown ``outcome``, empty cohort, no cohort/label intersection,
        or degenerate (all-zero / all-one) label vector.
    """
    # Local imports keep the dependency graph clean: pipeline.py is imported
    # by sequences.py for the RICH_* constants, so importing sequences /
    # torch_train at module load would form a cycle. Function-scope imports
    # break the cycle and are paid only when this entry point is called.
    from icumodelstream.sequences import build_sequence_tensors
    from icumodelstream.torch_train import fit_sequence_model, prepare_split_tensors

    if outcome not in {"mortality", "los_gt_7d"}:
        raise ValueError(
            f"outcome must be 'mortality' or 'los_gt_7d', got {outcome!r}."
        )

    config_snapshot: dict[str, Any] = {
        "cohort_spec": asdict(cohort_spec),
        "window_hours": window_hours,
        "seed": seed,
        "outcome": outcome,
        "include_hospice": include_hospice,
        "los_threshold_hours": los_threshold_hours,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "dropout": dropout,
        "max_epochs": max_epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        # device=None means "let fit_sequence_model autodetect at fit time";
        # we record that intent honestly rather than resolving it here.
        "device": device,
    }

    cohort, waterfall = build_cohort_with_waterfall(tables, cohort_spec)
    if cohort.height == 0:
        raise ValueError(
            "Cohort is empty after applying CohortSpec filters; cannot run sequence baseline. "
            f"Waterfall: total={waterfall.total_hospitalizations}, "
            f"after_age={waterfall.after_age_filter}, "
            f"after_icu={waterfall.after_icu_filter}."
        )

    # Same outcome dispatch + rename pattern as run_baseline_pipeline so the
    # downstream code never branches on outcome name beyond this point.
    if outcome == "mortality":
        raw_labels = extract_mortality_labels(tables, include_hospice=include_hospice)
        labels = raw_labels.rename({"mortality": "outcome"})
    else:  # outcome == "los_gt_7d"
        raw_labels = extract_los_label(tables, threshold_hours=los_threshold_hours)
        labels = raw_labels.rename({"long_los": "outcome"})

    cohort_id_dtype = cohort.schema["hospitalization_id"]
    label_id_dtype = labels.schema["hospitalization_id"]
    if cohort_id_dtype != label_id_dtype:
        raise ValueError(
            f"hospitalization_id dtype mismatch between cohort ({cohort_id_dtype}) and "
            f"labels ({label_id_dtype}); cast both sides to a common type before joining."
        )
    cohort_with_labels = cohort.join(labels, on="hospitalization_id", how="inner")
    if cohort_with_labels.height == 0:
        raise ValueError(
            "No cohort hospitalizations matched extracted labels; cannot run sequence baseline."
        )

    y_sum = int(cohort_with_labels["outcome"].sum())
    y_len = cohort_with_labels.height
    if y_sum == 0 or y_sum == y_len:
        raise ValueError(
            f"Label vector is degenerate (sum={y_sum}, n={y_len}); cannot fit sequence baseline. "
            "Need at least one example of each class."
        )

    anchors = get_admission_anchors(tables).join(
        cohort_with_labels.select("hospitalization_id"),
        on="hospitalization_id",
        how="inner",
    )

    sequences = build_sequence_tensors(
        tables,
        cohort_with_labels,
        anchors,
        window_hours=window_hours,
        vital_categories=RICH_VITAL_CATEGORIES,
        lab_categories=RICH_LAB_CATEGORIES,
        assessment_categories=RICH_ASSESSMENT_CATEGORIES,
        respiratory_devices=RICH_RESPIRATORY_DEVICES,
    )

    labels_for_torch = cohort_with_labels.select("hospitalization_id", "outcome")
    groups_for_torch = cohort_with_labels.select("hospitalization_id", "patient_id")

    # ---- Recover prevalences / split sizes from a one-off prepare_split_tensors call.
    # fit_sequence_model calls prepare_split_tensors internally with the same seed,
    # so this duplicates the split computation. The duplication is intentional and
    # cheap (just reshape + index): it keeps the split logic in one place
    # (prepare_split_tensors) without forcing a refactor to expose val/test
    # prevalences on SequenceResult. The split is deterministic under same seed
    # so both call sites produce identical folds.
    split_tensors = prepare_split_tensors(
        sequences, labels_for_torch, groups_for_torch, seed=seed
    )
    n_train = int(split_tensors.X_train.shape[0])
    n_val = int(split_tensors.X_val.shape[0])
    n_test = int(split_tensors.X_test.shape[0])
    train_prevalence = (
        float(split_tensors.y_train.mean().item()) if n_train > 0 else 0.0
    )
    val_prevalence = (
        float(split_tensors.y_val.mean().item()) if n_val > 0 else 0.0
    )
    test_prevalence = (
        float(split_tensors.y_test.mean().item()) if n_test > 0 else 0.0
    )
    # Recover the per-row test identity by indexing back into the source
    # sequences.hospitalization_ids array. Same split is used downstream by
    # fit_sequence_model (deterministic under same seed).
    test_hospitalization_ids = [
        str(sequences.hospitalization_ids[i]) for i in split_tensors.test_indices
    ]

    lstm_model, lstm_result = fit_sequence_model(
        sequences,
        labels_for_torch,
        groups_for_torch,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )

    return SequencePipelineResult(
        cohort_waterfall=waterfall,
        n_channels=len(sequences.channel_names),
        channel_names=list(sequences.channel_names),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        n_train_patients=split_tensors.n_train_patients,
        n_val_patients=split_tensors.n_val_patients,
        n_test_patients=split_tensors.n_test_patients,
        train_prevalence=train_prevalence,
        val_prevalence=val_prevalence,
        test_prevalence=test_prevalence,
        lstm=lstm_result,
        lstm_model=lstm_model,
        config_snapshot=config_snapshot,
        test_hospitalization_ids=test_hospitalization_ids,
        warnings=[],
    )


SUPPORTED_SUBGROUP_COLS: tuple[str, ...] = (
    "sex",
    "race_category",
    "ethnicity",
    "age_band",
    "icu_type",
)


def extract_subgroup_labels(
    tables: dict[str, TableRef],
    hospitalization_ids: list[str],
    subgroup_cols: list[str],
) -> pl.DataFrame:
    """Per-hospitalization subgroup labels joined from patient + hospitalization + adt.

    Parameters
    ----------
    tables:
        Discovered CLIF parquet tables.
    hospitalization_ids:
        Test-fold hospitalization_ids whose labels we need.
    subgroup_cols:
        Subset of ``SUPPORTED_SUBGROUP_COLS``: ``sex``, ``race_category``,
        ``ethnicity``, ``age_band``, ``icu_type``. Unknown values raise
        ``ValueError`` so the CLI fails loudly on typos.

    Returns
    -------
    polars.DataFrame with one row per input ``hospitalization_id`` (order
    preserved), and one column per requested subgroup plus ``hospitalization_id``.
    Null subgroup values stay null -- the consumer
    (:func:`icumodelstream.subgroups.compute_subgroup_metrics`) buckets them
    into ``"Unknown"`` per the no-silent-drop contract.

    Raises
    ------
    ValueError
        On unknown subgroup column names. Missing CLIF columns / tables raise
        from the underlying scan_table calls per CLAUDE.md rule 7.
    """
    unknown = set(subgroup_cols) - set(SUPPORTED_SUBGROUP_COLS)
    if unknown:
        raise ValueError(
            f"Unknown subgroup column(s) {sorted(unknown)}. "
            f"Supported: {list(SUPPORTED_SUBGROUP_COLS)}."
        )

    # Preserve caller order by starting with a positional id frame.
    base = pl.DataFrame(
        {"hospitalization_id": hospitalization_ids, "__row_idx": list(range(len(hospitalization_ids)))}
    )

    # Patient-level columns: join via hospitalization.patient_id
    needs_patient = any(c in subgroup_cols for c in ("sex", "race_category", "ethnicity"))
    if needs_patient:
        hosp = (
            scan_table(tables, "hospitalization")
            .select("hospitalization_id", "patient_id")
            .collect()
        )
        patient_cols = ["patient_id"]
        if "sex" in subgroup_cols:
            patient_cols.append("sex_category")
        if "race_category" in subgroup_cols:
            patient_cols.append("race_category")
        if "ethnicity" in subgroup_cols:
            patient_cols.append("ethnicity_category")
        patient = scan_table(tables, "patient").select(*patient_cols).collect()
        base = base.join(hosp, on="hospitalization_id", how="left").join(
            patient, on="patient_id", how="left"
        )
        # Rename to the subgroup keys the caller asked for.
        renames = {}
        if "sex" in subgroup_cols:
            renames["sex_category"] = "sex"
        if "ethnicity" in subgroup_cols:
            renames["ethnicity_category"] = "ethnicity"
        if renames:
            base = base.rename(renames)

    # age_band: from hospitalization.age_at_admission via subgroups.assign_age_band
    if "age_band" in subgroup_cols:
        from icumodelstream.subgroups import assign_age_band as _assign

        ages_df = (
            scan_table(tables, "hospitalization")
            .select("hospitalization_id", "age_at_admission")
            .collect()
        )
        base = base.join(ages_df, on="hospitalization_id", how="left")
        ages_np = base["age_at_admission"].to_numpy()
        # Polars converts null Int to null Python objects when calling to_numpy()
        # with object dtype; force object to preserve None for assign_age_band.
        ages_obj = base["age_at_admission"].to_list()
        base = base.with_columns(
            pl.Series("age_band", _assign(ages_obj).tolist(), dtype=pl.Utf8)
        )

    # icu_type: first adt.location_category per hospitalization (sorted by in_dttm)
    if "icu_type" in subgroup_cols:
        adt = scan_table(tables, "adt")
        adt_cols = set(adt.collect_schema().names())
        ts_col = "in_dttm" if "in_dttm" in adt_cols else None
        if ts_col is not None:
            icu = (
                adt.select("hospitalization_id", "location_category", ts_col)
                .sort([ts_col])
                .group_by("hospitalization_id")
                .agg(pl.col("location_category").first().alias("icu_type"))
                .collect()
            )
        else:
            icu = (
                adt.select("hospitalization_id", "location_category")
                .group_by("hospitalization_id")
                .agg(pl.col("location_category").first().alias("icu_type"))
                .collect()
            )
        base = base.join(icu, on="hospitalization_id", how="left")

    # Restore caller order and keep only the requested columns.
    final_cols = ["hospitalization_id"] + [c for c in subgroup_cols]
    return base.sort("__row_idx").select(final_cols)
