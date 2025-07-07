# Contributing to RAG Pipeline

Thank you for your interest in contributing to the RAG Pipeline project! 🎉

## 🚀 Quick Start

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create** a feature branch
4. **Make** your changes
5. **Test** your changes
6. **Submit** a pull request

## 🛠️ Development Setup

### Docker Development (Recommended)
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/rag-project.git
cd rag-project

# Start development environment
docker-compose -f docker-compose.yml -f docker-compose.override.yml up
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Start services
ollama serve
uvicorn src.api:app --reload
```

## 📝 Code Style

- Follow **PEP 8** for Python code
- Use **meaningful variable names**
- Add **docstrings** for functions and classes
- Keep functions **focused and small**

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/
```

## 🐛 Bug Reports

When reporting bugs, please include:
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **Environment details** (OS, Python version, etc.)

## 💡 Feature Requests

- Check if the feature already exists
- Explain the **use case**
- Provide **examples** if possible

## 📋 Pull Request Guidelines

- **One feature per PR**
- **Clear commit messages**
- **Update documentation** if needed
- **Add tests** for new functionality

## 🔄 Areas for Contribution

- **Bug fixes** and improvements
- **New document formats** support
- **Performance optimizations**
- **UI/UX enhancements**
- **Documentation** improvements
- **Test coverage** expansion

## 🤝 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## 📞 Questions?

- Open an **issue** for questions
- Check existing **discussions**
- Review the **README** first

---

**Happy coding!** 🚀 