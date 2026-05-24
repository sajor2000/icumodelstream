# CLIF-MIMIC Data Dictionary — Observed Schema

Snapshot of the 15 CLIF-MIMIC parquet tables as observed locally on 2026-05-24.
Total: **15 tables, 146,341,076 rows.** All files are named with the `clif_` prefix on disk;
the table key after `discover_tables` strips the prefix (so `clif_patient.parquet` → `"patient"`).

Authoritative spec: [CLIF 2.1 data dictionary](https://clif-icu.com/data-dictionary/data-dictionary-2.1.0).
This document captures the *as-observed* schema, including extras and quirks. When the local
schema disagrees with the CLIF 2.1 spec, both are recorded.

**Datetime convention:** all `*_dttm` columns are `Datetime(us, tz=UTC)` except `patient.birth_date`
which is `Datetime(us, tz=None)` (naive). Time-window filters in `features.py` should account
for that mismatch — see the U3 unit test in
`docs/plans/2026-05-24-002-feat-lightgbm-baseline-phase4-plan.md`.

**Identifier dtype:** `patient_id` and `hospitalization_id` are `String`, not `Int64`. Cohort
splits and label joins must respect that.

---

## Core tables (CLIF required)

### `patient` — 364,627 rows, 11 cols

| Column | Dtype | Notes |
|---|---|---|
| `patient_id` | String | Primary key |
| `race_name` / `race_category` | String | Dual representation |
| `ethnicity_name` / `ethnicity_category` | String | Dual representation |
| `sex_name` / `sex_category` | String | Dual representation |
| `birth_date` | Datetime(us, tz=None) | **Naive** — only tz-naive column in the dataset |
| `death_dttm` | Datetime(us, tz=UTC) | Alternative mortality signal — present if patient died, NULL otherwise |
| `language_name` / `language_category` | String | Dual representation |

### `hospitalization` — 546,028 rows, 17 cols

| Column | Dtype | Notes |
|---|---|---|
| `patient_id` | String | FK to patient |
| `hospitalization_id` | String | Primary key |
| `hospitalization_joined_id` | String | Linkage across joined stays |
| `admission_dttm` | Datetime(us, tz=UTC) | **Time anchor for Phase 4** |
| `discharge_dttm` | Datetime(us, tz=UTC) | End of stay |
| `age_at_admission` | Int64 | Already computed; one of `AGE_CANDIDATES` |
| `admission_type_name` / `admission_type_category` | String | Dual representation |
| `discharge_name` / `discharge_category` | String | **`discharge_category` holds the mortality signal** |
| `zipcode_*`, `census_*`, `state_code`, `county_code` | String | Geographic SDOH columns (CLIF spec extension) |

**`discharge_category` observed vocabulary** (12 values, all PHI-safe categories):

```
Acute Care Hospital, Acute Inpatient Rehab Facility,
Against Medical Advice (AMA), Assisted Living, Expired,
Home, Hospice, Long Term Care Hospital (LTACH),
Missing, Other, Psychiatric Hospital, Skilled Nursing Facility (SNF)
```

Mortality label: `discharge_category == "Expired"` (treat `"Hospice"` as a separate
opt-in via the `include_hospice` parameter in `labels.py`).

### `adt` — 1,458,408 rows, 9 cols

| Column | Dtype | Notes |
|---|---|---|
| `patient_id` / `hospitalization_id` / `hospital_id` | String | Identifiers |
| `in_dttm` / `out_dttm` | Datetime(us, tz=UTC) | Transfer interval |
| `location_name` / `location_category` / `location_type` | String | `location_category` is one of `ICU_TEXT_CANDIDATES`; cohort builder uses it to identify ICU stays |
| `hospital_type` | String | Hospital-level metadata |

---

## High-value optional tables

### `vitals` — 55,525,580 rows, 6 cols

| Column | Dtype | Notes |
|---|---|---|
| `hospitalization_id` | String | FK |
| `recorded_dttm` | Datetime(us, tz=UTC) | Time anchor for windowed aggregation |
| `vital_name` / `vital_category` | String | Dual representation |
| `vital_value` | Float64 | **In `VALUE_CANDIDATES` (now at priority 1)** |
| `meas_site_name` | String | Where vital was measured (sleeve cuff, etc.) |

### `labs` — 44,880,526 rows, 14 cols

| Column | Dtype | Notes |
|---|---|---|
| `hospitalization_id` | String | FK |
| `lab_order_dttm` / `lab_collect_dttm` / `lab_result_dttm` | Datetime(us, tz=UTC) | Three timestamps; Phase 4 uses `lab_result_dttm` for windowing (latest available) |
| `lab_order_name` | Null | All-null in this snapshot |
| `lab_order_category` | String | Order-side category |
| `lab_name` / `lab_category` | String | Dual representation |
| `lab_value` | String | **String dtype** — has units/qualifiers like `"<5"`, `"15 mg/dL"` |
| `lab_value_numeric` | Float64 | **The clean numeric — in `VALUE_CANDIDATES` at priority 2** |
| `reference_unit` | String | Unit for `lab_value_numeric` |
| `lab_specimen_name` / `lab_specimen_category` / `lab_loinc_code` | Null | All-null in this snapshot |

### `medication_admin_continuous` — 7,564,662 rows, 12 cols

| Column | Dtype | Notes |
|---|---|---|
| `hospitalization_id` / `med_order_id` | String | Identifiers |
| `med_name` / `med_category` / `med_group` | String | Three-level med taxonomy |
| `admin_dttm` | Datetime(us, tz=UTC) | Continuous infusion start |
| `mar_action_name` / `mar_action_category` | String | MAR (medication administration record) action |
| `med_dose` | Float32 | Infusion rate value |
| `med_dose_unit` | String | e.g., `"mcg/kg/min"` |
| `med_route_name` / `med_route_category` | String | Typically IV for continuous |

### `medication_admin_intermittent` — 3,252,010 rows, 12 cols

Same shape as `medication_admin_continuous`. `med_dose` is the per-administration dose
rather than an infusion rate.

### `respiratory_support` — 2,148,372 rows, 25 cols

Largest schema in the dataset (25 cols). Contains device mode, FiO2, PEEP, vent rate, etc.
See CLIF 2.1 spec for the full field list — too wide to enumerate here. Use
`notebooks/01_inspect.py` to view the live schema.

### `patient_assessments` — 19,389,145 rows, 8 cols

| Column | Dtype | Notes |
|---|---|---|
| `hospitalization_id` | String | FK |
| `recorded_dttm` | Datetime(us, tz=UTC) | Time anchor |
| `assessment_name` / `assessment_category` / `assessment_group` | String | Three-level assessment taxonomy (RASS, GCS, Braden, etc.) |
| `numerical_value` | Float64 | Numeric score (use this for windowed aggregation) |
| `categorical_value` | String | Categorical assessment result |

---

## Additional tables (CLIF spec extensions)

### `code_status` — 82,497 rows, 4 cols

| Column | Dtype |
|---|---|
| `patient_id` | String |
| `start_dttm` | Datetime(us, tz=UTC) |
| `code_status_name` / `code_status_category` | String |

Used for documenting DNR / Full Code orders. Note: keyed on `patient_id`, not
`hospitalization_id` — different from most other tables.

### `position` — 3,157,996 rows, 4 cols

Patient positioning records (prone, supine, etc.). Keyed on `hospitalization_id`.

### `crrt_therapy` — 467,609 rows, 11 cols

Continuous renal replacement therapy parameters. `device_id` and `dialysis_machine_name`
columns are all-null in this snapshot — drop or document depending on use case.

### `ecmo_mcs` — 93,399 rows, 10 cols

ECMO and mechanical circulatory support parameters. Lower-case field `fdO2` is a CLIF
spec quirk (most other columns are snake_case).

### `hospital_diagnosis` — 6,364,488 rows, 5 cols

| Column | Dtype | Notes |
|---|---|---|
| `hospitalization_id` | String | FK |
| `diagnosis_code` | String | ICD code |
| `diagnosis_code_format` | String | e.g., `"ICD10"`, `"ICD9"` |
| `diagnosis_primary` | Int32 | 1 if primary diagnosis, 0 otherwise |
| `poa_present` | Int32 | Present-on-admission flag |

### `patient_procedures` — 1,045,729 rows, 6 cols

Procedure codes per hospitalization. Less central to Phase 4 baseline (procedures are
intermediate causes, not exposures/features in a mortality model).

---

## Quirks and traps observed

1. **Null-dtype columns.** `labs.lab_order_name`, `labs.lab_specimen_*`, `labs.lab_loinc_code`,
   `crrt_therapy.device_id`, and `crrt_therapy.dialysis_machine_name` are all `Null` dtype
   (no values in this snapshot). Code that tries to read them must handle the Null dtype
   gracefully — polars accepts it but downstream operations like `cast` will fail loudly.

2. **String identifiers, not integers.** `patient_id` and `hospitalization_id` are `String`.
   Splits, joins, and merges must keep this dtype consistent — mixing `String` and `Int64`
   IDs in a join silently produces zero matches.

3. **Naive `birth_date` in an otherwise tz-aware schema.** If you need age-at-event,
   subtract `birth_date` from a UTC `_dttm` carefully — either localize `birth_date` to
   UTC first or strip tz from the other side. The existing `age_at_admission` column on
   `hospitalization` is the easier path.

4. **`lab_value` is a String.** This is intentional — it preserves qualifiers like `"<5"`
   or `"trace"`. For numeric work, always use `lab_value_numeric` (already in
   `VALUE_CANDIDATES`). Never `.cast(Float64)` the string column blindly.

5. **`code_status` is keyed by `patient_id`, not `hospitalization_id`.** Joining
   `code_status` to a per-hospitalization feature matrix requires a join on `patient_id`
   PLUS a time filter against `start_dttm` falling inside the hospitalization's
   `[admission_dttm, discharge_dttm]` interval.

6. **`vitals` is the largest behavioral table** (55.5M rows). Always use polars LazyFrame
   scans and filter early. Eager `pl.read_parquet` will use ~5 GB of RAM.

7. **`hospital_id` appears only in `adt`**, not in `hospitalization`. To get the admitting
   hospital for a hospitalization, join through `adt` on the first transfer row by
   `in_dttm`.

8. **`age_at_admission` is `Int64`, not `Float64`** — full integer years. Patients with
   ages above 89 are typically top-coded in MIMIC per PhysioNet rules; verify before
   stratifying by age band.

---

## Phase 4 baseline implications

The Phase 4 plan
(`docs/plans/2026-05-24-002-feat-lightgbm-baseline-phase4-plan.md`) builds on this snapshot:

- **Outcome (U2):** `discharge_category == "Expired"` is the mortality signal. The
  vocabulary `MORTALITY_VALUES = frozenset({"expired"})` (case-insensitive) covers the
  observed values. `include_hospice=True` would add `"hospice"`.
- **Time anchor (U3):** `hospitalization.admission_dttm` is the natural anchor for a
  baseline. ICU-admit time would require an `adt`-join (deferred).
- **Vitals timestamp (U3):** `vitals.recorded_dttm` is the only timestamp; `DATETIME_CANDIDATES`
  for vitals = `("recorded_dttm",)`.
- **Labs timestamp (U3):** Use `lab_result_dttm` for windowing — it is the latest of the
  three available timestamps, ensuring the lab result was actually back in time. CLIF spec
  is ambiguous; we pick result-time for the baseline and document it.
- **Identifier dtype (U4):** `patient_id` is String; the split helper must pass it through
  unchanged.
- **Class imbalance (U5):** Mortality prevalence at the hospitalization level can be
  computed once and cached; the Phase 4 plan budgets for `is_unbalance=True` in LightGBM
  to accommodate.

---

*Generated 2026-05-24 from local CLIF-MIMIC snapshot. Regenerate by running
`notebooks/01_inspect.py` and updating this file when the data root changes.*
