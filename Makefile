VENV = .venv
PY = $(VENV)/bin/python

.PHONY: install install-dl test lint format inspect qc cohort baseline notebook

$(VENV):
	python3.11 -m venv $(VENV)

notebook: $(VENV)
ifeq ($(strip $(NOTEBOOK)),)
	$(error NOTEBOOK is not set. Usage: make notebook NOTEBOOK=notebooks/01_inspect.py)
endif
	$(PY) -m marimo edit $(NOTEBOOK)

install: $(VENV)
	$(PY) -m pip install -e '.[dev,ml]'

install-dl: $(VENV)
	$(PY) -m pip install -e '.[dev,ml,ml-dl]'

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests

inspect:
	icumodelstream inspect --data-root $${CLIF_DATA_ROOT}

qc:
	mkdir -p reports
	icumodelstream qc --data-root $${CLIF_DATA_ROOT} --out reports/qc_summary.json

cohort:
	mkdir -p reports
	icumodelstream cohort --data-root $${CLIF_DATA_ROOT} --out reports/adult_icu_cohort.csv

baseline:
	mkdir -p reports models
	icumodelstream baseline --data-root $${CLIF_DATA_ROOT}
