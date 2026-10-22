import pandas as pd
from sklearn.model_selection import train_test_split

# The column we want to predict (the number of rings = the age)
TARGET = "Rings"


def load_data(path: str) -> pd.DataFrame:
    """Read the csv and drop the rows with a height of 0 (errors found in the EDA).

    Args:
        path: Path to the abalone csv file.

    Returns:
        The cleaned dataframe.
    """
    df = pd.read_csv(path)
    # An abalone can't have a height of 0, so we keep only the rows with a real height
    return df[df["Height"] > 0]


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into X_train, X_test, y_train, y_test (80/20).

    Args:
        df: The cleaned dataframe, with the target column.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    # X = the measurements (inputs), y = the rings (what we want to predict)
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    # 80% of the rows to train, 20% to test the model on data it never saw
    # random_state=42 gives the same split every time, so the results are reproducible
    return train_test_split(X, y, test_size=0.2, random_state=42)
