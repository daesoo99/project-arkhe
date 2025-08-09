.PHONY: test clean exp-entropy exp-demo exp-isolation

# Testing
test:
	pytest

test-unit:
	pytest tests/unit/

test-integration: 
	pytest tests/integration/

# Experiments (reproducible)
exp-entropy:
	python experiments/run_entropy_experiment.py --mode pilot

exp-demo:
	python experiments/run_quick_demo.py

exp-isolation:
	python src/orchestrator/isolation_pipeline.py

exp-baseline:
	python experiments/run_baseline_comparison.py

# Maintenance
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	rm -rf .pytest_cache/

clean-logs:
	rm -f logs/*.jsonl

# Code quality
format:
	black src/ tests/ experiments/
	isort src/ tests/ experiments/

lint:
	ruff check src/ tests/ experiments/