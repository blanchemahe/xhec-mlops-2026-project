from functools import lru_cache

import pandas as pd
from sklearn.pipeline import Pipeline

from src.modelling.step2_1_utils import load_pickle
from src.modelling.step3_predicting import predict
from src.web_service.lib.models import AbaloneInput, PredictionOutput

# API field name -> column name in the dataset (the model was trained with these names)
COLUMNS = {
    "sex": "Sex",
    "length": "Length",
    "diameter": "Diameter",
    "height": "Height",
    "whole_weight": "Whole weight",
    "shucked_weight": "Shucked weight",
    "viscera_weight": "Viscera weight",
    "shell_weight": "Shell weight",
}


@lru_cache
def get_model() -> Pipeline:
    """Load the model once (raises FileNotFoundError if the model was not trained).

    Returns:
        The trained pipeline.
    """
    # lru_cache keeps the result in memory, so the pickle is read only at the first request
    return load_pickle()


def run_inference(abalone: AbaloneInput) -> PredictionOutput:
    """Predict the rings and the age of one abalone.

    Args:
        abalone: The measurements sent to the API.

    Returns:
        The predicted rings and the age.
    """
    # One row dataframe, with the same column names as in training
    data = pd.DataFrame([abalone.model_dump()]).rename(columns=COLUMNS)
    rings = predict(get_model(), data)[0]
    # The age of an abalone is its number of rings + 1.5
    return PredictionOutput(rings=rings, age=rings + 1.5)
