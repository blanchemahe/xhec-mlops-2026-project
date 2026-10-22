import pandas as pd

from src.modelling.preprocessing import split_data
from src.modelling.training import evaluate_model, train_model
from src.modelling.utils import load_pickle, pickle_object


def make_fake_abalones(n: int = 100) -> pd.DataFrame:
    """Build a small fake dataset (the real one is not in git), where the rings grow with the weight."""
    return pd.DataFrame(
        {
            "Sex": ["M", "F", "I", "M"] * (n // 4),
            "Length": [0.3 + i * 0.004 for i in range(n)],
            "Diameter": [0.2 + i * 0.003 for i in range(n)],
            "Height": [0.05 + i * 0.001 for i in range(n)],
            "Whole weight": [0.2 + i * 0.02 for i in range(n)],
            "Shucked weight": [0.1 + i * 0.01 for i in range(n)],
            "Viscera weight": [0.05 + i * 0.005 for i in range(n)],
            "Shell weight": [0.07 + i * 0.006 for i in range(n)],
            "Rings": [5 + i // 6 for i in range(n)],
        }
    )


def test_train_evaluate_and_pickle(tmp_path) -> None:  # noqa: ANN001
    """Train on the fake data, check the metrics, then save and reload the model."""
    X_train, X_test, y_train, y_test = split_data(make_fake_abalones())
    model = train_model(X_train, y_train)

    assert evaluate_model(model, X_test, y_test)["rmse"] < 3

    pickle_object(model, tmp_path / "model.pkl")
    reloaded = load_pickle(tmp_path / "model.pkl")
    assert list(reloaded.predict(X_test)) == list(model.predict(X_test))
