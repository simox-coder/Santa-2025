# Santa 2025 Solver Makefile

.PHONY: setup download pull_kaggle_metric pull_kaggle_getting_started bench test tune solve validate score clean help

# Default environment variables
TOTAL_BUDGET_MIN ?= 180
MAX_WORKERS ?= auto
CLEAN_RUN ?= 0
COLLISION_BACKEND ?= auto

help:
	@echo "Santa 2025 - Christmas Tree Packing Solver"
	@echo ""
	@echo "Available targets:"
	@echo "  setup                  - Install dependencies"
	@echo "  download               - Download competition data from Kaggle"
	@echo "  pull_kaggle_metric     - Pull official metric notebook"
	@echo "  pull_kaggle_getting_started - Pull getting-started notebook"
	@echo "  bench                  - Run performance benchmarks"
	@echo "  test                   - Run unit tests"
	@echo "  tune                   - Run ASHA hyperparameter tuning"
	@echo "  solve                  - Generate submission from best config"
	@echo "  validate               - Validate submission file"
	@echo "  score                  - Compute official score"
	@echo "  clean                  - Clean generated artifacts"
	@echo ""
	@echo "Environment variables:"
	@echo "  TOTAL_BUDGET_MIN=${TOTAL_BUDGET_MIN} (default: 180)"
	@echo "  MAX_WORKERS=${MAX_WORKERS} (default: auto)"
	@echo "  CLEAN_RUN=${CLEAN_RUN} (default: 0)"
	@echo "  COLLISION_BACKEND=${COLLISION_BACKEND} (default: auto)"

setup:
	pip install -r requirements.txt
	@echo "Setup complete."

download:
	bash scripts/download.sh

pull_kaggle_metric:
	bash scripts/pull_notebooks.sh metric

pull_kaggle_getting_started:
	bash scripts/pull_notebooks.sh getting_started

bench:
	python scripts/bench_collision.py
	python scripts/bench_end2end.py
	@echo "Benchmarks complete. Results in artifacts/"

test:
	python -m pytest tests/ -v

tune:
	@if [ "$(CLEAN_RUN)" = "1" ]; then \
		rm -rf runs/*.jsonl artifacts/*.csv artifacts/*.yaml artifacts/*.json; \
	fi
	TOTAL_BUDGET_MIN=$(TOTAL_BUDGET_MIN) MAX_WORKERS=$(MAX_WORKERS) COLLISION_BACKEND=$(COLLISION_BACKEND) \
		python -m santa2025_solver tune

solve:
	python -m santa2025_solver solve

validate:
	python -m santa2025_solver.validate --submission artifacts/submission_best.csv

score:
	python -m santa2025_solver.score --submission artifacts/submission_best.csv

clean:
	rm -rf runs/*.jsonl
	rm -rf artifacts/*.csv artifacts/*.yaml artifacts/*.json artifacts/*.txt artifacts/*.md
	rm -rf __pycache__ santa2025_solver/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	@echo "Cleaned artifacts and cache."

# Combined targets
all: setup download pull_kaggle_metric pull_kaggle_getting_started test bench tune validate score
	@echo "Full pipeline complete!"

run_like_kaggle: download pull_kaggle_metric pull_kaggle_getting_started tune validate score
	@echo "Kaggle-like run complete!"
