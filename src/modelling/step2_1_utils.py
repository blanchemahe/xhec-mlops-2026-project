import pickle
from pathlib import Path

# Where the trained model is saved. The API will read it from the same place
MODEL_PATH = Path("src/web_service/local_objects/model.pkl")


def pickle_object(obj: object, path: Path = MODEL_PATH) -> None:
    """Save an object (the model pipeline) in a pickle file.

    Args:
        obj: The object to save.
        path: Where to save it.
    """
    # Create the folder if it doesn't exist yet
    path.parent.mkdir(parents=True, exist_ok=True)
    # "wb" = write in binary, pickle turns the python object into bytes
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path = MODEL_PATH) -> object:
    """Load an object from a pickle file.

    Args:
        path: Where the pickle file is.

    Returns:
        The saved object.
    """
    # "rb" = read in binary, pickle rebuilds the python object from the bytes
    with path.open("rb") as f:
        return pickle.load(f)
