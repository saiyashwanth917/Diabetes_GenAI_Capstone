
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from src.data_preprocessing import (
    prepare_pima_data,
    prepare_early_diabetes_data
)


# Project paths
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


# 1. Model comparison charts
def plot_model_comparison(dataset_name):

    file_path = (
        REPORT_DIR /
        f"{dataset_name}_model_comparison.csv"
    )

    df = pd.read_csv(file_path)

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    ax = df.set_index("Model")[metrics].plot(
        kind="bar",
        figsize=(11, 6),
        width=0.75
    )

    ax.set_title(
        f"{dataset_name.upper()} - Model Performance Comparison",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel("Score")
    ax.set_xlabel("Machine Learning Models")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()

    output = REPORT_DIR / f"{dataset_name}_model_comparison.png"

    plt.savefig(output, dpi=300)
    plt.close()

    print("Saved:", output)


# 2. Confusion matrix visualization
def plot_confusion_matrix(
    dataset_name,
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )

    plt.title(
        f"{dataset_name.upper()} - Confusion Matrix",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()

    output = REPORT_DIR / f"{dataset_name}_confusion_matrix.png"

    plt.savefig(output, dpi=300)
    plt.close()

    print("Saved:", output)


# Main function
def main():

    print("\nDIABETES MODEL VISUALIZATION")
    print("=" * 50)

    # Pima dataset
    print("\nGenerating Pima visualizations...")

    X_train, X_test, y_train, y_test, _ = prepare_pima_data()

    pima_model = joblib.load(
        MODEL_DIR / "pima_best_model.joblib"
    )

    plot_model_comparison("pima")

    plot_confusion_matrix(
        "pima",
        pima_model,
        X_test,
        y_test
    )

    # Early diabetes dataset
    print("\nGenerating Early Diabetes visualizations...")

    X_train, X_test, y_train, y_test, _ = (
        prepare_early_diabetes_data()
    )

    early_model = joblib.load(
        MODEL_DIR / "early_diabetes_best_model.joblib"
    )

    plot_model_comparison("early_diabetes")

    plot_confusion_matrix(
        "early_diabetes",
        early_model,
        X_test,
        y_test
    )

    print("\n" + "=" * 50)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("Check the reports folder.")
    print("=" * 50)


if __name__ == "__main__":
    main()
