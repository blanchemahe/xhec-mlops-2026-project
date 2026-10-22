# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

from src.modelling.preprocessing import load_data, split_data
from src.modelling.training import evaluate_model, train_model
from src.modelling.utils import pickle_object


def main(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle).

    Args:
        trainset_path: Path to the training csv.
    """
    # 1. Read and clean the data
    df = load_data(trainset_path)
    # 2. Split into train and test
    X_train, X_test, y_train, y_test = split_data(df)
    # 3. Train the model
    model = train_model(X_train, y_train)
    # 4. Check how good it is on the test set
    print("Test metrics:", evaluate_model(model, X_test, y_test))
    # 5. Save the model, the API will load it
    pickle_object(model)


if __name__ == "__main__":
    # Lets us run: uv run python -m src.modelling.main data/abalone.csv
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
