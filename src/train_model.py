# Model training functions

from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.data_preprocessing import (
    prepare_pima_data,
    prepare_early_diabetes_data
)


# Project folders
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# Machine learning algorithms
def get_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42
        ),

        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42
        )
    }


# Model training and evaluation
def train_and_evaluate(
    dataset_name,
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):

    print("\n" + "=" * 65)
    print(f"TRAINING MODELS: {dataset_name}")
    print("=" * 65)

    models = get_models()

    results = []
    trained_models = {}

    # Cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Train and compare models
    for model_name, classifier in models.items():

        print(f"\nTraining {model_name}...")

        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier)
        ])

        # Evaluate using training data only
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1
        )

        # Fit on complete training data
        pipeline.fit(X_train, y_train)

        # Predict test data
        predictions = pipeline.predict(X_test)

        # Evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        results.append({
            "Model": model_name,
            "CV F1 Score": cv_scores.mean(),
            "CV F1 Std": cv_scores.std(),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        trained_models[model_name] = pipeline

        print(f"Cross-validation F1: {cv_scores.mean():.4f}")
        print(f"Test Accuracy:       {accuracy:.4f}")
        print(f"Test Precision:      {precision:.4f}")
        print(f"Test Recall:         {recall:.4f}")
        print(f"Test F1 Score:       {f1:.4f}")

    # Create comparison table
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="CV F1 Score",
        ascending=False
    )

    print("\nMODEL COMPARISON")
    print("-" * 65)
    print(results_df.to_string(index=False))

    # Select best model based on cross-validation F1
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    # Save best model
    model_filename = (
        MODEL_DIR / f"{dataset_name.lower()}_best_model.joblib"
    )

    joblib.dump(best_model, model_filename)

    print("\nBEST MODEL:", best_model_name)
    print("Saved model:", model_filename)

    # Final classification report
    final_predictions = best_model.predict(X_test)

    report = classification_report(
        y_test,
        final_predictions,
        zero_division=0
    )

    print("\nCLASSIFICATION REPORT")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        final_predictions,
        labels=[0, 1]
    )

    print("CONFUSION MATRIX")
    print(cm)

    # Save comparison report
    report_filename = (
        REPORT_DIR / f"{dataset_name.lower()}_model_comparison.csv"
    )

    results_df.to_csv(report_filename, index=False)

    # Save classification report
    classification_filename = (
        REPORT_DIR / f"{dataset_name.lower()}_classification_report.txt"
    )

    with open(classification_filename, "w", encoding="utf-8") as file:
        file.write(f"Dataset: {dataset_name}\n")
        file.write(f"Best model: {best_model_name}\n\n")
        file.write(report)
        file.write("\nConfusion Matrix:\n")
        file.write(str(cm))

    print("\nReports saved successfully!")

    return results_df


# Main execution
def main():

    print("\nDIABETES GENAI CAPSTONE")
    print("MACHINE LEARNING MODEL TRAINING")

    # Dataset 1: Pima
    X_train, X_test, y_train, y_test, preprocessor = (
        prepare_pima_data()
    )

    pima_results = train_and_evaluate(
        "pima",
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    # Dataset 2: Early Stage Diabetes
    X_train, X_test, y_train, y_test, preprocessor = (
        prepare_early_diabetes_data()
    )

    early_results = train_and_evaluate(
        "early_diabetes",
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    print("\n" + "=" * 65)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 65)

    print("\nSaved models in:", MODEL_DIR)
    print("Saved reports in:", REPORT_DIR)


if __name__ == "__main__":
    main()
