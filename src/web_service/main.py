# Code with FastAPI (app = FastAPI(...))

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.modelling.step1_preprocessing import load_data
from src.modelling.step2_1_utils import METRICS_PATH
from src.web_service.lib.inference import COLUMNS, get_model, run_inference
from src.web_service.lib.models import AbaloneInput, PredictionOutput

app = FastAPI(
    title="Abalone age prediction",
    description="Predicts the number of rings (and the age) of an abalone from its measurements.",
)

DATA_PATH = "data/abalone.csv"
DASHBOARD = Path(__file__).parent / "static" / "index.html"


@app.get("/")
def home() -> dict:
    """Health check, to see if the API is running."""
    return {"health_check": "App up and running!"}


# Pydantic checks the body of the request with AbaloneInput before our code runs
@app.post("/predict", response_model=PredictionOutput, status_code=201)
def predict(payload: AbaloneInput) -> PredictionOutput:
    """Predict the number of rings and the age of an abalone.

    Args:
        payload: The measurements of the abalone.

    Returns:
        The predicted rings and the age.

    Raises:
        HTTPException: 503 if the model was not trained yet.
    """
    try:
        return run_inference(payload)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Model not found, train it first.") from error


@app.get("/dashboard", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the dashboard (one html file, see static/index.html)."""
    return FileResponse(DASHBOARD)


@app.get("/metrics")
def metrics() -> list:
    """Return the metrics of every training (one per line in the jsonl file)."""
    if not METRICS_PATH.exists():
        return []
    return [json.loads(line) for line in METRICS_PATH.read_text().splitlines() if line]


@app.get("/model")
def model_info() -> dict:
    """Return what the dashboard needs to follow the model live.

    These are the coefficients of the linear regression and, for each measurement, a useful
    slider range (1st percentile, 99th percentile, median).

    Returns:
        The intercept, the coefficients and the ranges.

    Raises:
        HTTPException: 503 if the model was not trained yet.
    """
    try:
        model = get_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Model not found, train it first.") from error
    names = model.named_steps["encode"].get_feature_names_out()
    coefs = dict(zip(names, model.named_steps["model"].coef_, strict=True))
    df = load_data(DATA_PATH)
    fields = {k: c for k, c in COLUMNS.items() if k != "sex"}
    return {
        "intercept": float(model.named_steps["model"].intercept_),
        "sex": {sex: float(coefs[f"sex__Sex_{sex}"]) for sex in "MFI"},
        "coefficients": {k: float(coefs[f"remainder__{c}"]) for k, c in fields.items()},
        "ranges": {
            k: [float(df[c].quantile(0.01)), float(df[c].quantile(0.99)), float(df[c].median())] for k, c in fields.items()
        },
    }
