import pandas as pd
from sklearn.pipeline import Pipeline


def predict(model: Pipeline, data: pd.DataFrame) -> list[float]:
    """Predict the number of rings for new abalones.

    Args:
        model: The trained pipeline.
        data: The measurements, with the same columns as in training.

    Returns:
        The predicted number of rings, one per row.
    """
    # The pipeline encodes Sex by itself, tolist() gives a normal python list instead of a numpy array
    return model.predict(data).tolist()
