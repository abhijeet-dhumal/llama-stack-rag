# Contributing to RAG Pipeline

Thank you for your interest in contributing to the RAG Pipeline project! This document provides guidelines and instructions for contributing.

## 🎯 How to Contribute

We welcome contributions in the following areas:
- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🧪 Test coverage
- 🔧 Performance optimizations
- 🌐 Internationalization

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/rag-pipeline.git
cd rag-pipeline

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/rag-pipeline.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy pre-commit

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 🛠️ Development Workflow

### Code Style

We use these tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pre-commit**: Automated checks

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type check
mypy src/ --ignore-missing-imports

# Run all checks
pre-commit run --all-files
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_rag_pipeline.py -v

# Run specific test
pytest tests/test_rag_pipeline.py::test_document_ingestion -v
```

### Local Development

```bash
# Start Ollama (required for development)
ollama serve

# Start the API in development mode
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Or use the start script
./start.sh
```

## 📝 Commit Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(api): add batch document ingestion endpoint

fix(embedder): handle empty text input gracefully

docs(readme): update installation instructions

test(pipeline): add integration tests for query processing
```

## 🔄 Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main

```bash
# Sync with upstream
git fetch upstream
git rebase upstream/main

# Run tests
pytest

# Run code quality checks
pre-commit run --all-files
```

### 2. Create Pull Request

1. Push your branch to your fork
2. Create a Pull Request on GitHub
3. Fill out the PR template completely
4. Link related issues using keywords (e.g., "Fixes #123")

### 3. PR Review Process

- Maintainers will review your PR
- Address feedback and push new commits
- Once approved, your PR will be merged

## 🐛 Bug Reports

When reporting bugs, please include:

- **Environment**: OS, Python version, dependency versions
- **Steps to reproduce**: Clear, numbered steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Logs**: Relevant error messages or logs
- **Additional context**: Screenshots, configuration files

Use this template:

```markdown
**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.11.5]
- RAG Pipeline version: [e.g., 1.0.0]

**Steps to reproduce:**
1. Start the API server
2. Upload a PDF document
3. Query the document

**Expected behavior:**
Should return relevant answer with sources

**Actual behavior:**
Returns empty response

**Logs:**
```
[paste relevant logs here]
```

**Additional context:**
Document size: 50 pages
```

## 💡 Feature Requests

For feature requests, please include:

- **Problem**: What problem does this solve?
- **Solution**: Proposed solution
- **Alternatives**: Other solutions considered
- **Use case**: Real-world scenarios
- **Priority**: How important is this feature?

## 🏗️ Architecture Guidelines

### Code Organization

```
src/
├── __init__.py
├── api.py              # FastAPI application
├── rag_pipeline.py     # Core RAG logic
└── utils/              # Utility functions
    ├── __init__.py
    ├── document_processor.py
    ├── embeddings.py
    └── vector_store.py

tests/
├── __init__.py
├── test_api.py
├── test_rag_pipeline.py
└── fixtures/           # Test data
```

### Design Principles

1. **Modularity**: Keep components loosely coupled
2. **Testability**: Write testable code with dependency injection
3. **Error Handling**: Use proper exception handling and logging
4. **Documentation**: Document public APIs and complex logic
5. **Performance**: Consider memory and CPU usage

### Adding New Features

1. **Document Processors**: Extend `DocumentProcessor` for new file types
2. **Embedders**: Implement `EmbedderInterface` for new embedding models
3. **Vector Stores**: Implement `VectorStoreInterface` for new databases
4. **LLMs**: Extend `LLMInterface` for new language models

## 🔒 Security

- Never commit API keys, passwords, or sensitive data
- Use environment variables for configuration
- Report security vulnerabilities privately to the maintainers
- Follow secure coding practices

## 📚 Documentation

### Code Documentation

```python
def process_document(self, file_path: str) -> Dict[str, Any]:
    """Process a document and extract content.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Dictionary containing extracted text and metadata
        
    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If file doesn't exist
    """
```

### API Documentation

- Update OpenAPI/Swagger docs for new endpoints
- Include request/response examples
- Document error codes and responses

## 🌟 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

## 📞 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: Tag maintainers for urgent issues

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to RAG Pipeline! 🚀 