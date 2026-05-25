# ICU Model Stream Roadmap

The first phase should prove that CLIF-MIMIC parquet data can be read, validated, summarized, and transformed into a stable cohort on a local Mac. Later phases can add baseline models, GPU training, and clinical-review reports.

| Phase | Goal | Exit criteria | Status |
|---|---|---|---|
| 0 | Repository skeleton | Installable package, CLI, tests, CLAUDE.md. | ✅ Done |
| 1 | CLIF-MIMIC inspection | Table inventory and schema gap report run locally. | ✅ Done (`notebooks/01_inspect.py`, `docs/data_dictionary_notes.md`) |
| 2 | QC and cohort | Missingness/outlier report and adult ICU cohort CSV. | ✅ Done (`notebooks/02_qc.py`, `notebooks/03_cohort.py`, `cohorts.build_cohort_with_waterfall`) |
| 3 | Baseline features | Vitals/labs/meds aggregates joined to cohort. | ✅ Done (`notebooks/04_features.py`, `features.aggregate_numeric_table_windowed`) |
| 4 | Baseline model | Reproducible LightGBM benchmark with leakage checks. | ✅ Done 2026-05-24 (`icumodelstream baseline`, `notebooks/05_baseline.py`) |
| 5 | GPU training | Separate rented-GPU scripts after local pipeline is stable. | ⏳ Gated on Phase 4 reproducibility verified against real data. |

GPU work should not begin until phases 1 through 4 are repeatable. Phase 4 is now structurally
complete with leakage-safe time-windowed features, patient-aware splits, calibration check, and
a reproducible CLI command. The reproducibility gate (Phase 5 prerequisite) is satisfied once
`make baseline` produces stable metrics across runs against the same data snapshot.
