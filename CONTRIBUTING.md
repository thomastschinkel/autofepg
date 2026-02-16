# Contributing to AutoFE-PG

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/thomastschinkel/autofepg.git
cd autofepg
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
make dev
```

## Running Tests

```bash
make test
```

With coverage:

```bash
make test-cov
```

## Code Style

This project uses **Black** for formatting and **isort** for import sorting.

```bash
make format   # auto-format
make lint     # check style
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch:

   ```bash
   git checkout -b feature/my-feature
   ```
3. Write tests for your changes
4. Ensure all tests pass:

   ```bash
   make test
   ```
5. Format your code:

   ```bash
   make format
   ```
6. Commit with a clear message:

   ```bash
   git commit -m "Add feature X"
   ```
7. Push and open a Pull Request

## Adding a New Feature Generator

1. Create a class inheriting from `FeatureGenerator` in `autofepg/generators.py`.
2. Implement:

   * `fit_transform(df, y, fold_indices)`
   * `transform(df)`
3. If the generator uses the target `y`, it **must use out-of-fold computation**.
4. Register the generator in `FeatureCandidateBuilder.build()` in `autofepg/builder.py`.
5. Add tests in `tests/test_autofepg.py`.

## Reporting Issues

* Use the GitHub issue tracker
* Include a **minimal reproducible example**
* Specify your versions of:

  * Python
  * NumPy
  * pandas
  * scikit-learn
  * XGBoost

name: CI

on:
push:
branches: [main]
pull_request:
branches: [main]

jobs:
test:
runs-on: ubuntu-latest
strategy:
matrix:
python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]

```
steps:
  - uses: actions/checkout@v4

  - name: Set up Python ${{ matrix.python-version }}
    uses: actions/setup-python@v5
    with:
      python-version: ${{ matrix.python-version }}

  - name: Cache pip
    uses: actions/cache@v4
    with:
      path: ~/.cache/pip
      key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
      restore-keys: |
        ${{ runner.os }}-pip-

  - name: Install dependencies
    run: |
      python -m pip install --upgrade pip
      pip install -e ".[dev]"

  - name: Lint
    run: |
      flake8 autofepg/ tests/ --max-line-length=100 --ignore=E501,W503,E402

  - name: Run tests
    run: |
      pytest tests/ -v --tb=short --cov=autofepg --cov-report=xml

  - name: Upload coverage
    if: matrix.python-version == '3.11'
    uses: codecov/codecov-action@v4
    with:
      file: ./coverage.xml
      fail_ci_if_error: false
```
