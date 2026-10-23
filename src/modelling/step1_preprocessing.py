import pandas as pd
from sklearn.model_selection import train_test_split

# The column we want to predict (the number of rings = the age)
TARGET = "Rings"

# Above this height an abalone is an error (found in the EDA)
MAX_HEIGHT = 0.5
WEIGHT_PART_COLS = ["Shucked weight", "Viscera weight", "Shell weight"]


def get_erroneous_rows(df: pd.DataFrame) -> pd.Series:
    """Flag the physically impossible rows (found in the EDA).

    Args:
        df: The abalone dataframe.

    Returns:
        A boolean series, True for the rows to remove.
    """
    return (
        (df["Height"] <= 0)
        | (df["Height"] > MAX_HEIGHT)
        | (df["Diameter"] > df["Length"])
        | df[WEIGHT_PART_COLS].gt(df["Whole weight"], axis=0).any(axis=1)
    )


def load_data(path: str) -> pd.DataFrame:
    """Read the csv and drop the erroneous rows (see get_erroneous_rows).

    Args:
        path: Path to the abalone csv file.

    Returns:
        The cleaned dataframe.
    """
    df = pd.read_csv(path)
    return df.loc[~get_erroneous_rows(df)]


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
