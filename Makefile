# Makefile for Cryptocurrency Event Study Project

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
FLAKE8 := flake8

# Directories
CODE_DIR := code
TEST_DIR := tests
DATA_DIR := data
OUTPUT_DIR := outputs

.PHONY: help install install-dev test coverage format lint clean run security-check setup

# Default target
help:
	@echo "Available targets:"
	@echo "  make install       - Install project dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run tests"
	@echo "  make coverage      - Run tests with coverage report"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Run code linting"
	@echo "  make clean         - Remove temporary files"
	@echo "  make run           - Run the main analysis"
	@echo "  make security-check - Check for security issues"
	@echo "  make setup         - Complete setup (install + env file)"

# Install dependencies
install:
	$(PIP) install -r requirements.txt

# Install development dependencies
install-dev: install
	$(PIP) install pytest pytest-cov black flake8 pre-commit
	pre-commit install

# Run tests
test:
	$(PYTEST) $(TEST_DIR) -v

# Run tests with coverage
coverage:
	$(PYTEST) $(TEST_DIR) --cov=$(CODE_DIR) --cov-report=html --cov-report=term

# Format code
format:
	$(BLACK) $(CODE_DIR)
	$(BLACK) $(TEST_DIR)

# Lint code
lint:
	$(FLAKE8) $(CODE_DIR) --max-line-length=120 --extend-ignore=E203,W503
	$(FLAKE8) $(TEST_DIR) --max-line-length=120 --extend-ignore=E203,W503

# Clean temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist build *.egg-info

# Run main analysis
run:
	cd $(CODE_DIR) && $(PYTHON) run_event_study_analysis.py

# Security check
security-check:
	@echo "Checking for security issues..."
	@$(PYTHON) security_check.py

# Setup environment
setup: install
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from template. Please add your API key."; \
	else \
		echo ".env file already exists."; \
	fi
	@echo "Setup complete. Don't forget to add your API key to .env!"

# Quick test run
quick-test:
	cd $(CODE_DIR) && $(PYTHON) -c "from data_preparation import DataPreparation; print('Import successful')"

# Generate documentation
docs:
	@echo "Documentation generation not yet configured"
	@echo "Consider using Sphinx or MkDocs"

# Build distribution
build:
	$(PYTHON) setup.py sdist bdist_wheel

# Upload to PyPI (test)
upload-test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Upload to PyPI (production)
upload:
	twine upload dist/*
