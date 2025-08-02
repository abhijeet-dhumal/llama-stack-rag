# Feast RAG Pipeline - Poetry Makefile
.PHONY: help install install-dev install-test install-lint install-docs install-deploy
.PHONY: test test-cov lint format type-check security docs clean
.PHONY: run dev build deploy pre-commit

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install core dependencies"
	@echo "  install-dev  - Install all dependency groups"
	@echo "  install-test - Install test dependencies"
	@echo "  install-lint - Install linting dependencies"
	@echo "  install-docs - Install documentation dependencies"
	@echo "  install-deploy - Install deployment dependencies"
	@echo ""
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run all linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  type-check   - Run mypy type checking"
	@echo "  security     - Run bandit security checks"
	@echo ""
	@echo "  run          - Run the application"
	@echo "  dev          - Run in development mode"
	@echo "  clean        - Clean cache and build files"
	@echo "  pre-commit   - Run pre-commit hooks"

# Installation commands
install:
	poetry install --only=main

install-dev:
	poetry install --with=test,lint,dev,docs,deploy

install-test:
	poetry install --with=test

install-lint:
	poetry install --with=lint

install-docs:
	poetry install --with=docs

install-deploy:
	poetry install --with=deploy

# Development commands
test:
	poetry run pytest

test-cov:
	poetry run pytest --cov=src --cov-report=html --cov-report=term

lint: format type-check security
	@echo "✅ Code formatting and import sorting completed"
	@echo "Note: flake8 temporarily disabled in favor of Black formatting"

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

type-check:
	poetry run mypy src/

security:
	poetry run bandit -r src/ -f json -o bandit-report.json || true

# Application commands
run:
	poetry run uvicorn src.api:app --host 0.0.0.0 --port 8000

dev:
	poetry run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Utility commands
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

pre-commit:
	poetry run pre-commit run --all-files

# Documentation
docs:
	poetry run mkdocs serve

build-docs:
	poetry run mkdocs build
