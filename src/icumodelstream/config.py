from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    root: Path
    format: str = "parquet"
    table_glob: str = "*.parquet"


class OutputConfig(BaseModel):
    report_dir: Path = Path("reports")
    cohort_csv: Path = Path("reports/adult_icu_cohort.csv")


class CohortConfig(BaseModel):
    min_age: int = 18
    require_icu_location: bool = True


class SafetyConfig(BaseModel):
    allow_phi: bool = False
    commit_patient_level_outputs: bool = False


class AppConfig(BaseModel):
    project: str = "icumodelstream"
    phase: str = "clif_mimic_parquet_local"
    data: DataConfig
    outputs: OutputConfig = Field(default_factory=OutputConfig)
    cohort: CohortConfig = Field(default_factory=CohortConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML config file into a typed application config."""
    config_path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text())
    return AppConfig.model_validate(raw)
