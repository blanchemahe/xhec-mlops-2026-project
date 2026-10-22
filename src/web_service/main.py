# Code with FastAPI (app = FastAPI(...))

from fastapi import FastAPI, HTTPException

from src.web_service.lib.inference import run_inference
from src.web_service.lib.models import AbaloneInput, PredictionOutput

app = FastAPI(
    title="Abalone age prediction",
    description="Predicts the number of rings (and the age) of an abalone from its measurements.",
)


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
