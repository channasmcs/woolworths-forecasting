.PHONY: help install install-dev test lint format type-check run pipeline clean

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	pip install -r requirements.txt

install-dev:  ## Install runtime + dev dependencies
	pip install -e ".[dev]"

test:  ## Run the test suite with coverage
	pytest

lint:  ## Lint with ruff (check only)
	ruff check src tests app.py

format:  ## Format code with ruff
	ruff format src tests app.py
	ruff check --fix src tests app.py

type-check:  ## Run mypy static type checking
	mypy src

run:  ## Launch the Streamlit dashboard
	streamlit run app.py

pipeline:  ## Run the full data + training pipeline
	python run_pipeline.py

clean:  ## Remove caches and generated files
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
