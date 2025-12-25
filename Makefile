# Santa 2025 Solver Makefile

PYTHON = python
SRC = src/santa2025_solver

.PHONY: tune solve validate score bench clean

tune:
	TOTAL_BUDGET_MIN=180 $(PYTHON) -m santa2025_solver.tune

solve:
	TOTAL_BUDGET_MIN=30 $(PYTHON) -m santa2025_solver.solve

validate:
	$(PYTHON) -m santa2025_solver.validate_cli --submission artifacts/submission_best.csv

score:
	$(PYTHON) -m santa2025_solver.score --submission artifacts/submission_best.csv

score-sample:
	$(PYTHON) -m santa2025_solver.score --submission sample_submission.csv

score-best:
	$(PYTHON) -m santa2025_solver.score --submission submission_best_165.csv

bench:
	$(PYTHON) -m santa2025_solver.collision.bench

clean:
	rm -rf artifacts/*.csv artifacts/*.txt artifacts/*.md artifacts/*.yaml
	rm -rf runs/*
	rm -rf __pycache__ $(SRC)/__pycache__ $(SRC)/**/__pycache__

install:
	pip install numpy pandas pyyaml
