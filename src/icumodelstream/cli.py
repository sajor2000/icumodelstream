from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from icumodelstream.cohorts import CohortSpec, build_adult_icu_cohort
from icumodelstream.io import discover_tables, table_inventory
from icumodelstream.models import BaselineResult, save_model
from icumodelstream.pipeline import BaselinePipelineResult, run_baseline_pipeline
from icumodelstream.qc import build_qc_report, write_qc_report
from icumodelstream.schema import validate_table_contracts, validation_results_to_frame

app = typer.Typer(help="Local-first CLIF-MIMIC parquet pipeline.")
console = Console()


def _resolve_data_root(data_root: Path | None) -> Path:
    if data_root is None:
        raise typer.BadParameter("Pass --data-root or set CLIF_DATA_ROOT before running commands.")
    return data_root.expanduser().resolve()


def _discover(data_root: Path) -> dict:
    """Discover tables, converting common discovery errors into typer.BadParameter."""
    try:
        return discover_tables(data_root)
    except FileNotFoundError as e:
        raise typer.BadParameter(str(e)) from e
    except ValueError as e:
        raise typer.BadParameter(
            f"{e}\nRename or remove one of the conflicting parquet files."
        ) from e


@app.command()
def inspect(
    data_root: Path = typer.Option(..., help="Directory containing CLIF parquet files."),
) -> None:
    """Inventory parquet tables and validate minimal CLIF contracts."""
    tables = _discover(_resolve_data_root(data_root))
    inventory = table_inventory(tables)

    rich_table = Table(title="Discovered CLIF parquet tables")
    for column in inventory.columns:
        rich_table.add_column(column)
    for row in inventory.iter_rows(named=True):
        rich_table.add_row(*(str(row[column]) for column in inventory.columns))
    console.print(rich_table)

    validation = validation_results_to_frame(validate_table_contracts(tables))
    validation_table = Table(title="Minimal core contract validation")
    for column in validation.columns:
        validation_table.add_column(column)
    for row in validation.iter_rows(named=True):
        validation_table.add_row(*(str(row[column]) for column in validation.columns))
    console.print(validation_table)


@app.command()
def qc(
    data_root: Path = typer.Option(..., help="Directory containing CLIF parquet files."),
    out: Path = typer.Option(Path("reports/qc_summary.json"), help="Output JSON path."),
) -> None:
    """Write a JSON QC report for discovered CLIF parquet tables."""
    tables = _discover(_resolve_data_root(data_root))
    report = build_qc_report(tables)
    out_path = write_qc_report(report, out)
    console.print(f"Wrote QC report: {out_path}")


@app.command()
def cohort(
    data_root: Path = typer.Option(..., help="Directory containing CLIF parquet files."),
    out: Path = typer.Option(Path("reports/adult_icu_cohort.csv"), help="Output cohort CSV path."),
    min_age: int = typer.Option(18, help="Minimum age for adult cohort."),
    require_icu_location: bool = typer.Option(
        True, help="Require an ICU-like ADT location when possible."
    ),
) -> None:
    """Build the initial adult ICU cohort CSV."""
    tables = _discover(_resolve_data_root(data_root))
    df = build_adult_icu_cohort(
        tables, CohortSpec(min_age=min_age, require_icu_location=require_icu_location)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)
    console.print(f"Wrote cohort with {df.height} rows: {out}")


def _git_head_sha() -> str | None:
    """Return the git HEAD SHA of the icumodelstream package's repo, or None if unavailable.

    Resolves `cwd` to the package directory so the SHA always reflects icumodelstream's
    state, not whatever directory the CLI happened to be invoked from. Failure to
    resolve a SHA must never break the CLI run.
    """
    package_dir = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            cwd=package_dir,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _nan_to_none(obj: Any) -> Any:
    """Recursively convert NaN floats to None for strict-JSON output (RFC 8259)."""
    import math
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _baseline_result_to_dict(result: BaselineResult) -> dict[str, Any]:
    """Serialize a BaselineResult to JSON-friendly dict.

    Drops the per-row arrays (y_true, y_pred_proba) intentionally: they would
    bloat the JSON and, although probabilities + 0/1 labels are not direct PHI,
    keeping baseline reports row-free matches CLAUDE.md's data-safety stance.
    Keeps the aggregate metrics dict and the calibration table.
    """
    return {
        "metrics": dict(result.metrics),
        "calibration_table": result.calibration_table.to_dicts(),
    }


def _build_metrics_payload(
    *,
    data_root: Path,
    result: BaselinePipelineResult,
    code_version: str | None,
    generated_at: str,
) -> dict[str, Any]:
    """Assemble the structured JSON payload written to ``--metrics-out``.

    Records only the basename of ``data_root`` (not the full absolute path) — paths
    can leak usernames and site identifiers per CLAUDE.md data-safety rules.
    """
    snap = result.config_snapshot
    config = {
        "data_root_basename": data_root.name,
        "cohort_spec": snap["cohort_spec"],
        "window_hours": snap["window_hours"],
        "test_size": snap["test_size"],
        "seed": snap["seed"],
        "include_hospice": snap["include_hospice"],
        "feature_set": snap.get("feature_set", "basic"),
        "outcome": snap.get("outcome", "mortality"),
        "los_threshold_hours": snap.get("los_threshold_hours", 168.0),
    }
    return {
        "config": config,
        "cohort_waterfall": asdict(result.cohort_waterfall),
        "n_features": result.n_features,
        "feature_names": list(result.feature_names),
        "split": {
            "n_train": result.n_train,
            "n_test": result.n_test,
            "n_train_patients": result.n_train_patients,
            "n_test_patients": result.n_test_patients,
            "train_prevalence": result.train_prevalence,
            "test_prevalence": result.test_prevalence,
        },
        "models": {
            "lightgbm": _baseline_result_to_dict(result.lightgbm),
            "logistic": _baseline_result_to_dict(result.logistic),
        },
        "warnings": list(result.warnings),
        "generated_at": generated_at,
        "code_version": code_version,
    }


def _format_metric(value: float | None) -> str:
    """Render a float metric with 3 decimals or ``n/a`` when missing/NaN."""
    if value is None:
        return "n/a"
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if fv != fv:  # NaN check without importing math
        return "n/a"
    return f"{fv:.3f}"


def _render_calibration_md(title: str, rows: list[dict[str, Any]]) -> str:
    """Render a calibration reliability table as Markdown."""
    lines = [
        f"### {title}",
        "",
        "| Bin | Mean pred | Mean actual | Count |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {bin} | {mean_pred} | {mean_actual} | {count} |".format(
                bin=row.get("bin"),
                mean_pred=_format_metric(row.get("mean_pred")),
                mean_actual=_format_metric(row.get("mean_actual")),
                count=row.get("count"),
            )
        )
    return "\n".join(lines)


def _build_markdown_summary(
    *,
    payload: dict[str, Any],
    metrics_out: Path,
    model_out: Path,
) -> str:
    """Render the human-readable Markdown summary written to ``--summary-out``."""
    config = payload["config"]
    cohort_spec = config["cohort_spec"]
    waterfall = payload["cohort_waterfall"]
    split = payload["split"]
    lightgbm = payload["models"]["lightgbm"]["metrics"]
    logistic = payload["models"]["logistic"]["metrics"]
    feature_names = payload["feature_names"]
    warnings = payload["warnings"]

    age_col = waterfall.get("age_col_used") or "n/a"
    icu_col = waterfall.get("icu_location_col_used") or "n/a"

    lines: list[str] = []
    lines.append(f"# Baseline metrics --- {payload['generated_at']}")
    lines.append("")
    lines.append(f"**Data root:** `{config['data_root_basename']}` (basename only)")
    lines.append(f"**Code version:** `{payload['code_version']}`")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Param | Value |")
    lines.append("|---|---|")
    lines.append(f"| min_age | {cohort_spec['min_age']} |")
    lines.append(
        f"| require_icu_location | {str(cohort_spec['require_icu_location']).lower()} |"
    )
    lines.append(f"| window_hours | {config['window_hours']} |")
    lines.append(f"| test_size | {config['test_size']} |")
    lines.append(f"| seed | {config['seed']} |")
    lines.append(f"| include_hospice | {str(config['include_hospice']).lower()} |")
    lines.append(f"| feature_set | {config.get('feature_set', 'basic')} |")
    outcome = config.get("outcome", "mortality")
    lines.append(f"| outcome | {outcome} |")
    if outcome == "los_gt_7d":
        lines.append(f"| los_threshold_hours | {config.get('los_threshold_hours', 168.0)} |")
    lines.append("")
    lines.append("## Cohort waterfall")
    lines.append("")
    lines.append("| Step | n hospitalizations |")
    lines.append("|---|---|")
    lines.append(f"| All hospitalizations | {waterfall['total_hospitalizations']:,} |")
    lines.append(f"| After age >= {cohort_spec['min_age']} | {waterfall['after_age_filter']:,} |")
    lines.append(f"| After ICU location filter | {waterfall['after_icu_filter']:,} |")
    lines.append(f"| Final | {waterfall['final']:,} |")
    lines.append("")
    lines.append(f"(age column resolved: `{age_col}`; ICU column resolved: `{icu_col}`)")
    lines.append("")
    lines.append("## Features")
    lines.append("")
    feature_list = ", ".join(feature_names) if feature_names else "(none)"
    lines.append(
        f"{payload['n_features']} features from {waterfall['final']:,} hospitalizations: "
        f"{feature_list}"
    )
    lines.append("")
    lines.append("## Split")
    lines.append("")
    lines.append(
        f"- n_train = {split['n_train']:,} ({split['n_train_patients']:,} unique patients), "
        f"prevalence = {_format_metric(split['train_prevalence'])}"
    )
    lines.append(
        f"- n_test  = {split['n_test']:,} ({split['n_test_patients']:,} unique patients), "
        f"prevalence = {_format_metric(split['test_prevalence'])}"
    )
    lines.append("")
    lines.append("(Split is by patient_id; no patient appears in both folds.)")
    lines.append("")
    lines.append("## Model metrics")
    lines.append("")
    lines.append("| Model | AUROC | AUPRC | Brier | Calib intercept | Calib slope |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        "| LightGBM | {auroc} | {auprc} | {brier} | {intercept} | {slope} |".format(
            auroc=_format_metric(lightgbm.get("auroc")),
            auprc=_format_metric(lightgbm.get("auprc")),
            brier=_format_metric(lightgbm.get("brier_score")),
            intercept=_format_metric(lightgbm.get("calibration_intercept")),
            slope=_format_metric(lightgbm.get("calibration_slope")),
        )
    )
    lines.append(
        "| Logistic | {auroc} | {auprc} | {brier} | {intercept} | {slope} |".format(
            auroc=_format_metric(logistic.get("auroc")),
            auprc=_format_metric(logistic.get("auprc")),
            brier=_format_metric(logistic.get("brier_score")),
            intercept=_format_metric(logistic.get("calibration_intercept")),
            slope=_format_metric(logistic.get("calibration_slope")),
        )
    )
    lines.append("")
    lines.append("## Calibration (reliability tables)")
    lines.append("")
    lines.append(
        _render_calibration_md("LightGBM", payload["models"]["lightgbm"]["calibration_table"])
    )
    lines.append("")
    lines.append(
        _render_calibration_md("Logistic", payload["models"]["logistic"]["calibration_table"])
    )
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("None")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Metrics JSON: `{metrics_out}`")
    lines.append(
        f"- LightGBM model: `{model_out}` (load via `lightgbm.Booster(model_file=...)`)"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*Generated by `icumodelstream baseline`. Reproduce with the same seed and code_version.*"
    )
    return "\n".join(lines) + "\n"


def _print_baseline_terminal_summary(
    *,
    result: BaselinePipelineResult,
    metrics_out: Path,
    summary_out: Path,
    model_out: Path,
) -> None:
    """Print a one-screen Rich summary after a baseline run."""
    total_patients = result.n_train_patients + result.n_test_patients
    cohort_n = result.cohort_waterfall.final
    console.print("=== Baseline complete ===")
    console.print(
        f"Cohort:    {cohort_n:,} hospitalizations ({total_patients:,} unique patients)"
    )
    console.print(
        f"Features:  {result.n_features} (vitals + labs aggregates, "
        f"first {result.config_snapshot['window_hours']}h)"
    )
    console.print(
        f"Split:     {result.n_train:,} train / {result.n_test:,} test (by patient_id)"
    )
    outcome_label = {
        "mortality": "Mortality",
        "los_gt_7d": "LOS > 7d",
    }.get(result.config_snapshot.get("outcome", "mortality"), "Outcome")
    console.print(
        f"{outcome_label}: {result.train_prevalence:.1%} train, "
        f"{result.test_prevalence:.1%} test"
    )
    console.print("")
    lgbm = result.lightgbm.metrics
    logr = result.logistic.metrics
    console.print(
        "LightGBM  "
        f"AUROC {_format_metric(lgbm.get('auroc'))}  "
        f"AUPRC {_format_metric(lgbm.get('auprc'))}  "
        f"Brier {_format_metric(lgbm.get('brier_score'))}  "
        f"CalibSlope {_format_metric(lgbm.get('calibration_slope'))}"
    )
    console.print(
        "Logistic  "
        f"AUROC {_format_metric(logr.get('auroc'))}  "
        f"AUPRC {_format_metric(logr.get('auprc'))}  "
        f"Brier {_format_metric(logr.get('brier_score'))}  "
        f"CalibSlope {_format_metric(logr.get('calibration_slope'))}"
    )
    console.print("")
    console.print(f"Wrote {metrics_out}")
    console.print(f"Wrote {summary_out}")
    console.print(f"Wrote {model_out}")


@app.command()
def baseline(
    data_root: Path = typer.Option(..., help="Directory containing CLIF parquet files."),
    metrics_out: Path = typer.Option(
        Path("reports/baseline_metrics.json"), help="Output JSON path."
    ),
    summary_out: Path = typer.Option(
        Path("reports/baseline_summary.md"), help="Output Markdown summary path."
    ),
    model_out: Path = typer.Option(
        Path("models/baseline_lightgbm.txt"),
        help="LightGBM native-format model path.",
    ),
    min_age: int = typer.Option(18, help="Minimum age for adult cohort."),
    require_icu_location: bool = typer.Option(True, help="Require an ICU-like ADT location."),
    window_hours: float = typer.Option(
        24.0, help="Feature observation window in hours from admission."
    ),
    test_size: float = typer.Option(0.2, help="Holdout fraction (split by patient_id)."),
    seed: int = typer.Option(42, help="Random seed for split + model training."),
    include_hospice: bool = typer.Option(
        False, help="Treat 'Hospice' discharge as mortality=1."
    ),
    feature_set: str = typer.Option(
        "basic",
        help="Feature richness. 'basic' = 8 aggregate vitals+labs columns; "
        "'rich' = per-category vitals (HR, SBP, etc.) + per-category labs + GCS/RASS + "
        "respiratory device flags (~85 features).",
    ),
    outcome: str = typer.Option(
        "mortality",
        help="Outcome label to predict. 'mortality' = in-hospital death "
        "(default); 'los_gt_7d' = length of stay > threshold (default 168h = 7d).",
    ),
    los_threshold_hours: float = typer.Option(
        168.0,
        help="LOS threshold in hours when --outcome=los_gt_7d. Default 168 = 7 days.",
    ),
) -> None:
    """Run the Phase 4 LightGBM + logistic baseline and write metrics + model artifacts."""
    resolved_root = _resolve_data_root(data_root)
    tables = _discover(resolved_root)

    cohort_spec = CohortSpec(min_age=min_age, require_icu_location=require_icu_location)
    result = run_baseline_pipeline(
        tables,
        cohort_spec,
        window_hours=window_hours,
        test_size=test_size,
        seed=seed,
        include_hospice=include_hospice,
        feature_set=feature_set,
        outcome=outcome,
        los_threshold_hours=los_threshold_hours,
    )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    code_version = _git_head_sha()
    payload = _build_metrics_payload(
        data_root=resolved_root,
        result=result,
        code_version=code_version,
        generated_at=generated_at,
    )

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    metrics_out.write_text(
        json.dumps(_nan_to_none(payload), indent=2, sort_keys=False, allow_nan=False)
    )
    summary_out.write_text(
        _build_markdown_summary(
            payload=payload, metrics_out=metrics_out, model_out=model_out
        )
    )
    save_model(result.lightgbm_model, model_out)

    _print_baseline_terminal_summary(
        result=result,
        metrics_out=metrics_out,
        summary_out=summary_out,
        model_out=model_out,
    )


if __name__ == "__main__":
    app()
