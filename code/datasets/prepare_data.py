from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = ROOT / "data" / "raw" / "winequality-red.csv"
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
TEST_FILE = ROOT / "data" / "processed" / "test.csv"


def remove_outliers(df):
    numeric = df.select_dtypes(include="number").columns

    for column in numeric:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df = df[(df[column] >= lower) & (df[column] <= upper)]

    return df


def main():
    df = pd.read_csv(RAW_FILE, sep=";")

    df = df.drop_duplicates()
    df = df.dropna()
    df = remove_outliers(df)

    df["quality"] = (df["quality"] >= 6).astype(int)

    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["quality"],
    )

    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)

    print(f"Training data: {train.shape}")
    print(f"Testing data: {test.shape}")
    print(f"Saved training data to: {TRAIN_FILE}")
    print(f"Saved testing data to: {TEST_FILE}")


if __name__ == "__main__":
    main()