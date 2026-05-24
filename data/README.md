# Data directory

Do not commit data to this repository. Place local data outside the repository when possible. If you must use this directory for local experiments, use the ignored subdirectories below:

| Directory | Purpose | Git status |
|---|---|---|
| `raw/` | Original CLIF-MIMIC parquet files or symlinks. | Ignored |
| `interim/` | Temporary transformed files. | Ignored |
| `processed/` | Derived feature matrices or cohorts. | Ignored |
| `external/` | External metadata not safe for git. | Ignored |

Only this README should be committed.
