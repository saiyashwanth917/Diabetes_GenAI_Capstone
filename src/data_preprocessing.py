
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


# =====================================================
# 1. PIMA DIABETES DATA PREPROCESSING
# =====================================================

def prepare_pima_data(test_size=0.2, random_state=42):

    """
    Preprocess Pima Indians Diabetes dataset.
    """

    file_path = DATA_DIR / "pima_diabetes.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Pima dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    print("\nPreparing Pima Diabetes Dataset...")
    print("Original records:", len(df))

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove duplicate records
    df = df.drop_duplicates().copy()

    print("Records after removing duplicates:", len(df))

    # Separate features and target
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # Validate target values
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("Invalid values found in Pima Outcome column.")

    # Replace medically invalid zero measurements with NaN
    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    X[zero_columns] = X[zero_columns].replace(
        0, float("nan")
    )

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            X.columns.tolist()
        )
    ])

    # Split data before fitting preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("\nPima preprocessing completed!")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


# =====================================================
# 2. EARLY STAGE DIABETES DATA PREPROCESSING
# =====================================================

def prepare_early_diabetes_data(
    test_size=0.2,
    random_state=42
):

    """
    Preprocess UCI Early Stage Diabetes dataset.
    """

    file_path = DATA_DIR / "early_diabetes.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Early Diabetes dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    print("\nPreparing Early Stage Diabetes Dataset...")
    print("Original records:", len(df))

    # Clean column names
    df.columns = df.columns.str.strip()

    # Clean whitespace in categorical values
    for column in df.select_dtypes(
        include=["object"]
    ).columns:

        df[column] = df[column].str.strip()

    # Remove duplicate records
    original_count = len(df)

    df = df.drop_duplicates().copy()

    print("Records after removing duplicates:", len(df))
    print("Duplicates removed:", original_count - len(df))

    # Separate features and target
    X = df.drop(columns=["class"])

    y = (
        df["class"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "positive": 1,
            "negative": 0
        })
    )

    # Validate target labels
    if y.isna().any():
        raise ValueError(
            "Unexpected target labels found in class column."
        )

    # Identify numerical features
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Identify categorical features
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nNumerical features:", numerical_columns)
    print("Categorical features:", categorical_columns)

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    # Combine preprocessing pipelines
    transformers = []

    if numerical_columns:
        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("\nEarly Diabetes preprocessing completed!")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


# =====================================================
# 3. TEST PREPROCESSING FUNCTIONS
# =====================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("DIABETES GENAI CAPSTONE")
    print("DATA PREPROCESSING")
    print("=" * 55)

    # Test Pima dataset
    prepare_pima_data()

    # Test Early Diabetes dataset
    prepare_early_diabetes_data()

    print("\n" + "=" * 55)
    print("ALL PREPROCESSING FUNCTIONS EXECUTED SUCCESSFULLY!")
    print("=" * 55)
