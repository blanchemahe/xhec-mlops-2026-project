# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

from prefect import flow, task

from src.modelling.step1_preprocessing import load_data, split_data
from src.modelling.step2_1_utils import log_metrics, pickle_object
from src.modelling.step2_training import evaluate_model, train_model

# task(...) turns a normal function into a prefect task
# Prefect then shows each step (and if it worked) in the UI
load_data_task = task(load_data)
split_data_task = task(split_data)
train_model_task = task(train_model)
evaluate_model_task = task(evaluate_model)
pickle_object_task = task(pickle_object)
log_metrics_task = task(log_metrics)


# A flow is the whole pipeline, made of tasks. log_prints=True shows our prints in the Prefect logs
@flow(name="train-abalone-model", log_prints=True)
def main(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle).

    Args:
        trainset_path: Path to the training csv.
    """
    # 1. Read and clean the data
    df = load_data_task(trainset_path)
    # 2. Split into train and test
    X_train, X_test, y_train, y_test = split_data_task(df)
    # 3. Train the model
    model = train_model_task(X_train, y_train)
    # 4. Check how good it is on the test set
    metrics = evaluate_model_task(model, X_test, y_test)
    print("Test metrics:", metrics)
    # 5. Keep the metrics of this run, the dashboard shows the typical error
    log_metrics_task(metrics)
    # 6. Save the model, the API will load it
    pickle_object_task(model)


if __name__ == "__main__":
    # Lets us run: uv run python -m src.modelling.main data/abalone.csv
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
