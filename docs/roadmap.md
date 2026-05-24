# ICU Model Stream Roadmap

The first phase should prove that CLIF-MIMIC parquet data can be read, validated, summarized, and transformed into a stable cohort on a local Mac. Later phases can add baseline models, GPU training, and clinical-review reports.

| Phase | Goal | Exit criteria |
|---|---|---|
| 0 | Repository skeleton | Installable package, CLI, tests, CLAUDE.md. |
| 1 | CLIF-MIMIC inspection | Table inventory and schema gap report run locally. |
| 2 | QC and cohort | Missingness/outlier report and adult ICU cohort CSV. |
| 3 | Baseline features | Vitals/labs/meds aggregates joined to cohort. |
| 4 | Baseline model | Reproducible LightGBM benchmark with leakage checks. |
| 5 | GPU training | Separate rented-GPU scripts after local pipeline is stable. |

GPU work should not begin until phases 1 through 4 are repeatable.
