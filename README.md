# ICU Model Stream

ICU Model Stream is the first implementation scaffold for **CLIF-Navigator** using **CLIF-MIMIC parquet data**. The initial goal is intentionally narrow: read credentialed CLIF parquet tables, validate the table layout, build a reproducible ICU cohort, generate quality-control reports, and create baseline features before any paid GPU training is attempted.

This repository is designed for local development on a Mac, including an **M4 Pro with 64 GB RAM**. The Mac should be used to build and validate the data pipeline. Large neural model training should happen later on a rented CUDA GPU only after local tests pass.

## What this repo does first

The first milestone is not a bedside clinical model. It is a clean, reproducible data pipeline that answers four questions: can the CLIF parquet files be discovered, can the expected tables be read, can we validate basic schema and missingness, and can we construct a stable adult ICU cohort for modeling.

| Layer | Purpose | First implementation |
|---|---|---|
| Data I/O | Discover and read CLIF parquet tables without loading everything into memory. | `src/icumodelstream/io.py` |
| Schema checks | Confirm expected CLIF table/column presence and report gaps. | `src/icumodelstream/schema.py` |
| QC | Summarize rows, columns, missingness, time ranges, and candidate outliers. | `src/icumodelstream/qc.py` |
| Cohort | Build a reproducible adult ICU cohort from patient, hospitalization, and ADT-like tables. | `src/icumodelstream/cohorts.py` |
| Features | Create simple baseline aggregates for later LightGBM and neural models. | `src/icumodelstream/features.py` |
| CLI | Run inspect, QC, and cohort jobs from the terminal. | `src/icumodelstream/cli.py` |

## Quick start

Create a Python environment and install the project in editable mode.

```bash
cd icumodelstream
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev,ml]'
```

Copy the example config and edit the path to point to your local CLIF-MIMIC parquet directory. Do **not** commit real data.

```bash
cp configs/local.example.yaml configs/local.yaml
# edit configs/local.yaml
```

Run the first local checks.

```bash
icumodelstream inspect --data-root /path/to/clif_mimic_parquet
icumodelstream qc --data-root /path/to/clif_mimic_parquet --out reports/qc_summary.json
icumodelstream cohort --data-root /path/to/clif_mimic_parquet --out reports/adult_icu_cohort.csv
```

## Data safety

This repository must not contain MIMIC credentials, local hospital PHI, parquet files, CSV extracts, model checkpoints trained on PHI, or screenshots with patient-level rows. The `.gitignore` file blocks common data paths, but responsible review is still required before every commit.

## Development principle

The code should stay simple until the data pipeline is proven. Prefer small verified functions, deterministic tests, explicit schemas, and command-line workflows that a clinical researcher can reproduce.
