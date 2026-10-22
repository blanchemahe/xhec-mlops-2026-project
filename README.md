# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)

Predict the age of abalones (number of shell rings) from simple physical measurements, instead of cutting, staining and counting rings under a microscope.

## Team

| Name | GitHub |
|---|---|
| Lancelot Brisset | [@brissetlancelot92](https://github.com/brissetlancelot92) |
| Louan Bardou | [@louan-username](https://github.com/louan-username) |
| Blanche Mahé | [@blanchemahe](https://github.com/blanchemahe) |
| Alix Bernal | [@alix-username](https://github.com/alix-username) |

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.10 or 3.11 (uv installs Python 3.11 automatically, as pinned in `.python-version`)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/blanchemahe/xhec-mlops-2026-project.git
cd xhec-mlops-2026-project

# Create the virtual environment (.venv) and install all dependencies from uv.lock
uv sync

# Install the pre-commit hooks (ruff formatting/linting, notebook output stripping)
uv run pre-commit install
```

To run any command inside the environment, prefix it with `uv run` (e.g. `uv run python ...`), or activate it with `source .venv/bin/activate`.

### Dataset

Download the [Abalone dataset](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) into the `data/` folder (ignored by git):

```bash
mkdir -p data
curl -L -o data/abalone.zip https://www.kaggle.com/api/v1/datasets/download/rodolfomendes/abalone-dataset
unzip -o data/abalone.zip -d data && rm data/abalone.zip
```

You should end up with `data/abalone.csv` (4,177 rows, 9 columns).

## Development

### Managing dependencies

```bash
uv add <package>          # app dependency
uv add --dev <package>    # development-only dependency
```

`uv add` updates `pyproject.toml` and `uv.lock`; commit both files.

### Code quality

Pre-commit hooks run automatically on each `git commit`. To run them manually on the whole repository:

```bash
uv run pre-commit run --all-files
```

The same checks run in the CI (GitHub Actions) on every push and pull request.
