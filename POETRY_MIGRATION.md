# Poetry Migration Guide

This document explains the transition from `requirements.txt` to Poetry for dependency management.

## 🎯 Benefits of Poetry

- **Dependency Groups**: Organize dependencies by purpose (test, lint, dev, docs, deploy)
- **Lock File**: `poetry.lock` ensures reproducible builds
- **Virtual Environment**: Built-in virtual environment management
- **Build System**: Integrated packaging and distribution
- **Simplified Commands**: Easy dependency management and task running

## 📦 Dependency Groups

### Main Dependencies (`[tool.poetry.dependencies]`)
- **Core Application**: FastAPI, Pydantic, Uvicorn
- **RAG Pipeline**: Ollama, PyMilvus, NumPy, Pandas
- **Document Processing**: PyPDF2, python-docx
- **ML Libraries**: sentence-transformers, scikit-learn
- **Feast**: feast[milvus], marshmallow<4.0.0

### Test Dependencies (`[tool.poetry.group.test.dependencies]`)
- **Testing Framework**: pytest, pytest-asyncio
- **Coverage**: pytest-cov
- **Mocking**: pytest-mock

### Lint Dependencies (`[tool.poetry.group.lint.dependencies]`)
- **Code Formatting**: black, isort
- **Linting**: flake8, mypy
- **Security**: bandit
- **Documentation**: pydocstyle
- **Pre-commit**: pre-commit

### Development Dependencies (`[tool.poetry.group.dev.dependencies]`)
- **Interactive Development**: ipython, jupyter, notebook

### Documentation Dependencies (`[tool.poetry.group.docs.dependencies]`)
- **Static Site Generator**: mkdocs, mkdocs-material
- **API Documentation**: mkdocstrings[python]

### Deployment Dependencies (`[tool.poetry.group.deploy.dependencies]`)
- **Production Server**: gunicorn
- **Containerization**: docker

## 🚀 Common Commands

### Installation
```bash
# Install only main dependencies
poetry install --only=main

# Install with specific groups
poetry install --with=test,lint

# Install all dependencies
poetry install --with=test,lint,dev,docs,deploy
```

### Development Workflow
```bash
# Run tests
poetry run pytest
# or
make test

# Format code
poetry run black src/ tests/
poetry run isort src/ tests/
# or
make format

# Run linting
poetry run flake8 src/
# or
make lint

# Run application
poetry run uvicorn src.api:app --reload
# or
make dev
```

### Dependency Management
```bash
# Add a new dependency
poetry add fastapi

# Add a development dependency
poetry add --group dev jupyter

# Add a test dependency
poetry add --group test pytest-benchmark

# Remove a dependency
poetry remove package-name

# Update dependencies
poetry update

# Show dependency tree
poetry show --tree
```

### Environment Management
```bash
# Activate virtual environment
poetry shell

# Run command in environment
poetry run python script.py

# Show environment info
poetry env info

# Create new environment
poetry env use python3.12
```

## 🔄 Migration Steps

1. **Backup Current Setup**
   ```bash
   cp requirements.txt requirements.txt.backup
   ```

2. **Install Poetry**
   ```bash
   pip install poetry
   ```

3. **Initialize Poetry** (Done)
   ```bash
   poetry init
   ```

4. **Install Dependencies**
   ```bash
   poetry install --with=test,lint
   ```

5. **Test Setup**
   ```bash
   poetry run pytest
   ```

6. **Update CI/CD** (if applicable)
   - Replace `pip install -r requirements.txt` with `poetry install`
   - Use `poetry run` for running commands

## 📝 Configuration Files

- **`pyproject.toml`**: Main configuration file containing dependencies and tool settings
- **`poetry.lock`**: Lock file ensuring reproducible installs (commit to git)
- **`Makefile`**: Convenient commands for common tasks

## 🔧 Tool Configurations

All tool configurations are now centralized in `pyproject.toml`:

- **Black**: Code formatting settings
- **isort**: Import sorting configuration
- **Pytest**: Test discovery and markers
- **MyPy**: Type checking configuration
- **Coverage**: Test coverage settings

## 🆚 Before vs After

### Before (requirements.txt)
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest
black src/
flake8 src/
```

### After (Poetry)
```bash
poetry install --with=test,lint
make test
make format
make lint
```

## 🎯 Best Practices

1. **Commit Poetry Files**: Always commit `pyproject.toml` and `poetry.lock`
2. **Use Groups**: Organize dependencies into logical groups
3. **Lock Dependencies**: Use `poetry lock` to update lock file
4. **Environment Isolation**: Use `poetry shell` or `poetry run`
5. **Semantic Versioning**: Use appropriate version constraints

## 🚨 Troubleshooting

### Common Issues

1. **Poetry not found**: Ensure Poetry is installed and in PATH
2. **Lock file conflicts**: Run `poetry lock --no-update` to regenerate
3. **Version conflicts**: Use `poetry show --outdated` to check updates
4. **Virtual environment**: Use `poetry env info` to check environment status

### Migration from pip

If you have an existing `requirements.txt`:
```bash
# Install from requirements.txt
poetry add $(cat requirements.txt)

# Or manually add each dependency
poetry add package-name==version
```
