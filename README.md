# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)

<a href="https://louanbardou.github.io/xhec-mlops-2026-project/"><img src="docs/dashboard-cropped.jpg" alt="Dashboard" width="50%"></a>

**Live demo: https://louanbardou.github.io/xhec-mlops-2026-project/**

Predict the age of abalones (number of shell rings) from simple physical measurements, instead of cutting, staining and counting rings under a microscope.

## Team

| Name | GitHub |
|---|---|
| Lancelot Brisset | [@brissetlancelot92](https://github.com/brissetlancelot92) |
| Louan Bardou | [@louanbardou](https://github.com/louanbardou) |
| Blanche Mahé | [@blanchemahe](https://github.com/blanchemahe) |
| Alix Bernal | [@ALX-7777](https://github.com/ALX-7777) |

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

# Create the virtual environment (.venv), install all dependencies from uv.lock and the pre-commit hooks
make install
```

`make train` trains the model, `make api` starts the API and the dashboard, `make test` runs the tests.

To run any command inside the environment, prefix it with `uv run` (e.g. `uv run python ...`), or activate it with `source .venv/bin/activate`.

### Dataset

The [Abalone dataset](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) is in the repository: `data/abalone.csv` (4,177 rows, 9 columns).

## Notebooks (EDA and modelling)

The analysis behind the model is in `notebooks/`:

- `eda.ipynb`: data quality checks (9 physically impossible rows found), target and feature analysis, and the decisions taken for the modelling.
- `modelling.ipynb`: baseline, linear regression and random forest compared by cross-validation, with every run tracked in MLflow. The linear regression is selected and registered as `abalone-rings-regressor` with the `champion` alias.

Open them in VS Code or Jupyter with the `.venv` kernel, after downloading the dataset. The MLflow runs are stored locally in `mlflow.db` and `mlruns/` (ignored by git). To compare them, run from the project root, then open http://localhost:5001:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

## Train the model

```bash
uv run python -m src.modelling.main data/abalone.csv
```

The model is saved in `src/web_service/local_objects/model.pkl` (ignored by git). The code is in `src/modelling/`: `step1_preprocessing.py` (load + split), `step2_training.py` (model + metrics), `step2_1_utils.py` (save and load the model), `step3_predicting.py` (predict with the saved model), `main.py` (runs everything).
The model is a linear regression, with `Sex` one-hot encoded inside a sklearn pipeline.

Run the tests with `uv run pytest`.

## Prefect (automatic retraining)

`main.py` is a Prefect flow (one task per step), so `uv run python -m src.modelling.main data/abalone.csv` runs it once. To retrain every Monday at 3am, use 2 terminals:

```bash
uv run prefect server start          # terminal 1: prefect server, UI on http://127.0.0.1:4200
uv run python -m src.modelling.deploy   # terminal 2: creates the "weekly-retraining" deployment (keep it running)
```

Open http://127.0.0.1:4200 to see the deployment and the flow runs. To test without waiting for Monday, click "Run" on the deployment in the UI, or in a 3rd terminal:
`uv run prefect deployment run 'train-abalone-model/weekly-retraining'`

`fastapi` is capped below 0.151 because newer versions break the Prefect server.

## API (FastAPI + Docker)

`POST /predict` returns the rings and the age (rings + 1.5) of one abalone. Inputs are validated with Pydantic.

```bash
# Without Docker (train the model first)
uv run uvicorn src.web_service.main:app --port 8001   # http://127.0.0.1:8001/docs

# With Docker (the model is trained at build)
docker build -f Dockerfile.app -t abalone-app .
docker run -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-app   # API :8000, Prefect UI :4200
```

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"sex": "M", "length": 0.455, "diameter": 0.365, "height": 0.095, "whole_weight": 0.514, "shucked_weight": 0.2245, "viscera_weight": 0.101, "shell_weight": 0.15}'
```

## Dashboard

An ocean scene served by the API: move the sliders (left) and the abalone grows ring by ring up to its predicted age. The "Age drivers" panel (right) shows how much each measurement counts in the linear model.

```bash
uv run uvicorn src.web_service.main:app --port 8001   # train first, then open http://127.0.0.1:8001/dashboard
```

With Docker: http://localhost:8000/dashboard. A copy that needs no server (used by GitHub Pages) is built into `docs/index.html` with `uv run python -m bin.build_static`.

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
