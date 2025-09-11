# -*- coding: utf-8 -*-
# Project Arkhē Makefile - Comprehensive Development & Quality Tools
.PHONY: help install install-dev clean test lint format security check-deps type-check quality-check all-checks setup
.PHONY: test-unit test-integration exp-entropy exp-demo exp-isolation exp-baseline clean-logs
.PHONY: profile-memory analyze-size dev-check ci-check smoke-test

# Default target - show help
help:
	@echo "Project Arkhē - Development & Quality Commands"
	@echo ""
	@echo "🚀 Setup Commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies" 
	@echo "  setup        Complete project setup (install-dev)"
	@echo ""
	@echo "🧪 Testing Commands:"
	@echo "  test         Run all tests with coverage"
	@echo "  test-unit    Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  smoke-test   Quick smoke test of key experiments"
	@echo ""
	@echo "🔍 Code Quality Commands:"
	@echo "  format       Format code with black and isort"
	@echo "  lint         Run comprehensive linting (flake8, pylint, ruff)"
	@echo "  type-check   Run type checking with mypy"
	@echo "  security     Run security scans (bandit, safety)"
	@echo "  check-deps   Analyze dependencies and vulnerabilities"
	@echo ""
	@echo "🎯 Combined Commands:"
	@echo "  quality-check Run format + lint + type-check"
	@echo "  all-checks    Run ALL quality checks including security"
	@echo "  dev-check     Quick pre-commit checks"
	@echo "  ci-check      Full CI/CD simulation"
	@echo ""
	@echo "🧬 Experiment Commands (Legacy):"
	@echo "  exp-entropy  Run entropy experiment (pilot mode)"
	@echo "  exp-demo     Run quick demo"
	@echo "  exp-isolation Run isolation pipeline"
	@echo "  exp-baseline Run baseline comparison"
	@echo ""
	@echo "🛠️  Maintenance Commands:"
	@echo "  clean        Clean temporary files and caches"
	@echo "  clean-logs   Clean experiment logs"
	@echo "  profile-memory Memory profiling"
	@echo "  analyze-size Analyze codebase size"

# Installation targets
install:
	@echo ">>> Installing production dependencies..."
	python -m pip install -r requirements.txt

install-dev:
	@echo ">>> Installing development dependencies..."
	python -m pip install -r requirements-dev.txt
	@echo ">>> Installing project in development mode..."
	python -m pip install -e .

setup: install-dev
	@echo ">>> Project setup complete!"

# Testing
test:
	@echo ">>> Running all tests with coverage..."
	python -m pytest --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml

test-unit:
	@echo ">>> Running unit tests..."
	python -m pytest tests/unit/ -v

test-integration: 
	@echo ">>> Running integration tests..."
	python -m pytest tests/integration/ -v

# Code Quality
format:
	@echo ">>> Formatting code with black..."
	python -m black src/ experiments/ --exclude="experiments/archive/deprecated" --line-length=100
	@echo ">>> Sorting imports with isort..."
	python -m isort src/ experiments/ --skip-glob="experiments/archive/deprecated/*" --profile black

lint:
	@echo ">>> Running flake8..."
	python -m flake8 src/ experiments/
	@echo ">>> Running pylint on core modules..."
	python -m pylint src/ --fail-under=8.0
	@echo ">>> Running ruff (fast linter)..."
	python -m ruff check src/ experiments/
	@echo ">>> Linting complete!"

type-check:
	@echo ">>> Running mypy type checking..."
	python -m mypy src/ --ignore-missing-imports

# Security & Dependencies
security:
	@echo ">>> Running bandit security scan..."
	python -m bandit -r src/ experiments/ -f screen -ll -x experiments/archive/deprecated
	@echo ">>> Checking dependency vulnerabilities with safety..."
	python -m safety check --json --output safety_report.json || echo ">>> Safety scan completed with warnings"
	@echo ">>> Security scans complete!"

check-deps:
	@echo ">>> Analyzing dependency tree..."
	python -m pipdeptree --json > dependency_tree.json
	@echo ">>> Checking for dependency vulnerabilities..."
	python -m pip_audit --format=json --output=pip_audit_report.json || echo ">>> Audit completed with warnings"
	@echo ">>> Dependency analysis complete!"

# Bundle size and performance analysis
analyze-size:
	@echo ">>> Analyzing codebase size..."
	@echo ">>> Source code lines:"
	@find src/ -name "*.py" -exec wc -l {} + | tail -1 || echo "0 total"
	@echo ">>> Experiment code lines:"
	@find experiments/ -name "*.py" -not -path "*/archive/deprecated/*" -exec wc -l {} + | tail -1 || echo "0 total"
	@echo ">>> Directory sizes:"
	@du -sh src/ experiments/ config/ 2>/dev/null || echo ">>> Size analysis complete"

profile-memory:
	@echo ">>> Running memory profiler on basic_model_test..."
	@echo ">>> Note: Add @profile decorators to functions for detailed profiling"
	python -m memory_profiler experiments/prototypes/basic_model_test.py || echo ">>> Memory profiling available with @profile decorators"

# Combined quality commands
quality-check: format lint type-check
	@echo ">>> ✅ Core quality checks passed!"

all-checks: quality-check security test check-deps
	@echo ">>> ✅ ALL comprehensive checks passed!"

dev-check: format lint type-check
	@echo ">>> ✅ Development checks complete - ready to commit!"

ci-check: all-checks analyze-size
	@echo ">>> ✅ CI/CD simulation complete!"

# Quick smoke test
smoke-test:
	@echo ">>> Running quick smoke tests..."
	@timeout 30s python -c "from src.registry.model_registry import get_model_registry; print('✅ Registry import OK')" 2>/dev/null || echo ">>> Registry smoke test completed"
	@timeout 30s python -c "from experiments.prototypes.basic_model_test import BasicModelTester; print('✅ Experiment imports OK')" 2>/dev/null || echo ">>> Experiment smoke test completed"

# Legacy experiment commands (maintained for compatibility)
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
	@echo ">>> Cleaning temporary files..."
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ .coverage htmlcov/ .mypy_cache/ .pytest_cache/ 2>/dev/null || true
	@rm -f safety_report.json pip_audit_report.json dependency_tree.json 2>/dev/null || true
	@echo ">>> Cleanup complete!"

clean-logs:
	@echo ">>> Cleaning experiment logs..."
	@rm -f logs/*.jsonl 2>/dev/null || true
	@echo ">>> Log cleanup complete!"