# Contributing to NexTune

Thank you for your interest in contributing to NexTune! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)

---

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect differing viewpoints and experiences

---

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/NexTune.git
cd NexTune
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov hypothesis black flake8
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

---

## Development Workflow

### Project Structure

- `app.py` — Main Streamlit application
- `src/` — Source code modules
- `scripts/` — Utility scripts
- `notebooks/` — Jupyter notebooks for analysis
- `tests/` — Test suite
- `data/` — Datasets
- `models/` — Trained model artifacts

### Making Changes

1. **Write Code** — Implement your feature or fix
2. **Add Tests** — Write tests for new functionality
3. **Run Tests** — Ensure all tests pass
4. **Format Code** — Use Black for formatting
5. **Lint Code** — Check with flake8
6. **Commit** — Write clear commit messages

---

## Coding Standards

### Python Style Guide

- Follow **PEP 8** style guide
- Use **Black** for code formatting (line length: 88)
- Use **type hints** where appropriate
- Write **docstrings** for functions and classes

### Example

```python
def predict_price(
    brand: str,
    rating: float,
    has_anc: bool,
    bt_version: float
) -> float:
    """
    Predict headphone price based on specifications.
    
    Args:
        brand: Brand name
        rating: Product rating (1.0-5.0)
        has_anc: Whether product has ANC
        bt_version: Bluetooth version
        
    Returns:
        Predicted price in INR
    """
    # Implementation
    pass
```

### Code Formatting

```bash
# Format code with Black
black app.py src/ scripts/

# Check linting
flake8 app.py src/ scripts/
```

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py -v

# Run property-based tests
pytest tests/ -v -k "property"
```

### Writing Tests

- Write unit tests for new functions
- Use property-based testing (Hypothesis) for data validation
- Aim for >80% code coverage
- Test edge cases and error conditions

### Example Test

```python
import pytest
from hypothesis import given, strategies as st

def test_price_prediction():
    """Test basic price prediction."""
    result = predict_price("Sony", 4.5, True, 5.3)
    assert 500 <= result <= 50000

@given(
    rating=st.floats(min_value=1.0, max_value=5.0),
    bt_version=st.floats(min_value=4.0, max_value=6.0)
)
def test_price_prediction_property(rating, bt_version):
    """Property-based test for price prediction."""
    result = predict_price("Sony", rating, True, bt_version)
    assert isinstance(result, float)
    assert result > 0
```

---

## Submitting Changes

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add battery life prediction feature
fix: Correct scrolling issue in Streamlit app
docs: Update installation instructions
test: Add property tests for data preprocessing
refactor: Simplify feature engineering pipeline
```

### Pull Request Process

1. **Update Documentation** — Update README if needed
2. **Add Tests** — Ensure new code is tested
3. **Run All Tests** — Make sure everything passes
4. **Create PR** — Write a clear description of changes
5. **Address Feedback** — Respond to review comments

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Manual testing completed

## Checklist
- [ ] Code follows PEP 8
- [ ] Documentation updated
- [ ] No breaking changes
```

---

## Areas for Contribution

### High Priority

- 🐛 **Bug Fixes** — Fix reported issues
- 📝 **Documentation** — Improve docs and examples
- ✅ **Tests** — Increase test coverage
- 🎨 **UI/UX** — Enhance Streamlit interface

### Feature Ideas

- Add more ML models (XGBoost, LightGBM)
- Implement API endpoint (FastAPI)
- Add data visualization dashboard
- Support for more e-commerce platforms
- Real-time price tracking
- Price trend analysis

### Good First Issues

Look for issues labeled `good-first-issue` or `help-wanted` on GitHub.

---

## Questions?

- Open an issue on GitHub
- Contact the maintainer: [ESpoorthy](https://github.com/ESpoorthy)

---

Thank you for contributing to NexTune! 🎧
