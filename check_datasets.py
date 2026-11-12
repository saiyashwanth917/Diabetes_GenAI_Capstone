
import pandas as pd
from pathlib import Path

# Dataset paths
PIMA_PATH = Path("data/raw/pima_diabetes.csv")
EARLY_PATH = Path("data/raw/early_diabetes.csv")


def inspect_dataset(path, dataset_name, target):
    print("\n" + "=" * 60)
    print(f"{dataset_name} INSPECTION")
    print("=" * 60)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    df = pd.read_csv(path)

    print("\nDataset shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five records:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate records:", df.duplicated().sum())

    if target in df.columns:
        print("\nTarget distribution:")
        print(df[target].value_counts(dropna=False))
    else:
        print(f"\nERROR: Target column '{target}' not found.")

    print("\nStatistical summary:")
    print(df.describe(include="all").T)

    return df


if __name__ == "__main__":
    pima = inspect_dataset(
        PIMA_PATH,
        "PIMA INDIANS DIABETES",
        "Outcome"
    )

    early = inspect_dataset(
        EARLY_PATH,
        "UCI EARLY STAGE DIABETES",
        "class"
    )
