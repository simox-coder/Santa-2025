.PHONY: setup download pull_kaggle_metric pull_kaggle_getting_started test tune solve score validate clean

# Environment variables with defaults
TOTAL_BUDGET_MIN ?= 180
MAX_WORKERS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

download:
	@echo "Downloading competition data..."
	@$(PYTHON) -m santa2025_solver.download

pull_kaggle_metric:
	@echo "Pulling Kaggle metric notebook..."
	@$(PYTHON) -m santa2025_solver.pull_notebooks metric

pull_kaggle_getting_started:
	@echo "Pulling Kaggle getting started notebook..."
	@$(PYTHON) -m santa2025_solver.pull_notebooks getting_started

pull_notebooks: pull_kaggle_metric pull_kaggle_getting_started

test:
	$(PYTHON) -m pytest tests/ -v

tune:
	@echo "Running orchestrator with ASHA tuning..."
	@echo "TOTAL_BUDGET_MIN=$(TOTAL_BUDGET_MIN) MAX_WORKERS=$(MAX_WORKERS)"
	TOTAL_BUDGET_MIN=$(TOTAL_BUDGET_MIN) MAX_WORKERS=$(MAX_WORKERS) $(PYTHON) -m santa2025_solver.orchestrator

solve:
	@echo "Running solver with best config..."
	$(PYTHON) -m santa2025_solver.solve

score:
	@echo "Scoring submission..."
	$(PYTHON) -m santa2025_solver.score --submission artifacts/submission_best.csv

validate:
	@echo "Validating submission..."
	$(PYTHON) -m santa2025_solver.validate --submission artifacts/submission_best.csv

all: setup download pull_notebooks tune score

clean:
	rm -rf artifacts/*.csv artifacts/*.yaml artifacts/*.txt
	rm -rf runs/*.jsonl
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
