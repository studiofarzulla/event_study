# Contributing to Cryptocurrency Event Study

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing. It's fairly generic and straightforward and if anyone does work on this, I salute you my friend. 

## Code of Conduct

By participating in this project, you agree to abide by our principles of respectful and constructive collaboration (no roasting me in the discussion though feel free to email me with it).

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/event_study.git
   cd event_study
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/event_study.git
   ```
4. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up pre-commit hooks** (recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names (please use my style)
- Maximum line length: 120 characters
- Use type hints where appropriate (more the merrier)
- Format code with `black`:
  ```bash
  black code/
  ```

### Documentation

- Add docstrings to all functions, classes, and modules
- Use Google-style docstrings format
- Update README.md if adding new features
- Include examples in docstrings where helpful

### Testing

- Write tests for new functionality
- Ensure all tests pass before submitting PR:
  ```bash
  pytest tests/ -v
  ```
- Aim for >80% code coverage for new code
- Test edge cases and error conditions

### Commit Messages

Follow conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks
- (or just use GitHub Copilot)

Example:
```
feat: add support for additional cryptocurrencies

- Added SOL and MATIC to supported coins
- Updated API fetcher to handle new symbols
- Added tests for new functionality
```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:
- Python version
- Operating system
- Complete error message and stack trace
- Minimal code example to reproduce the issue
- Expected vs. actual behavior
- If you manage to run it all. 

### Feature Requests

For feature requests, please describe:
- The problem you're trying to solve (check out my other iteration, there was more problems than solutions)
- Your proposed solution
- Alternative solutions you've considered (I've considered like 10 different methodologies without hitting API/tech limits, I shall accept graciously other workarounds)
- Any relevant examples or use cases

## Submitting Pull Requests

1. **Update your fork**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Make your changes** following the guidelines above

3. **Run tests**:
   ```bash
   pytest tests/
   ```

4. **Update documentation** if needed

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub with:
   - Clear title and description
   - Reference to any related issues
   - Summary of changes made
   - Screenshots if applicable

## Code Review Process

1. Maintainers (me) will review your PR within 5-7 days
2. Address any requested changes
3. Once approved, your PR will be merged (code buddies)

## Areas for Contribution

### Priority Areas
- [ ] Additional statistical tests
- [ ] Performance optimizations
- [ ] Support for more cryptocurrencies
- [ ] Interactive visualizations
- [ ] API error handling improvements
- [ ] Documentation improvements


## Recognition

Contributors will be acknowledged in:
- The project README
- Release notes
- Project documentation
- Poster in my study.

Thank you for contributing to my overcomplicated yet beautiful creation!
