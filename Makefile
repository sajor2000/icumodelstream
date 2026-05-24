.PHONY: install test lint format inspect qc cohort

install:
	python3.11 -m pip install -e '.[dev,ml]'

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
