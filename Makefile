.PHONY: setup download test bench tune solve validate score export clean

setup:
	pip install -r requirements.txt

download:
	@echo "Note: Kaggle credentials required for download"
	@echo "Set KAGGLE_USERNAME and KAGGLE_KEY environment variables"
	@echo "Or place kaggle.json in ~/.kaggle/"
	-kaggle competitions download -c santa-2025 -p data/raw
	-unzip -o data/raw/santa-2025.zip -d data/raw 2>/dev/null || true

test:
	python -m pytest tests/ -v

bench:
	python scripts/bench_collision.py

tune:
	python -m santa2025_solver.tune

solve:
	python -m santa2025_solver.solve

validate:
	python -m santa2025_solver.validate

score:
	python -m santa2025_solver.score

export:
	python -m santa2025_solver.export

clean:
	rm -rf artifacts/* runs/*
	rm -f exports/*
