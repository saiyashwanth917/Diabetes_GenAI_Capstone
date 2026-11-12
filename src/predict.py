
from pathlib import Path

import joblib
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


# Load trained models
def load_pima_model():

    model_path = MODEL_DIR / "pima_best_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "Pima model not found. Run python -m src.train_model first."
        )

    return joblib.load(model_path)


def load_early_diabetes_model():

    model_path = MODEL_DIR / "early_diabetes_best_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "Early Diabetes model not found. Run python -m src.train_model first."
        )

    return joblib.load(model_path)


# Common prediction function
def generate_prediction(model, input_data):

    # Get the exact feature order used during training
    feature_names = list(model.feature_names_in_)

    # Check required input fields
    missing_features = [
        feature for feature in feature_names
        if feature not in input_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing input features: {missing_features}"
        )

    # Create input DataFrame
    input_df = pd.DataFrame(
        [[input_data[name] for name in feature_names]],
        columns=feature_names
    )

    # Generate prediction
    prediction = int(model.predict(input_df)[0])

    # Calculate positive-class probability score
    probabilities = model.predict_proba(input_df)[0]

    positive_index = list(model.classes_).index(1)
    probability_score = float(probabilities[positive_index])

    result = {
        "prediction": prediction,
        "result": (
            "Positive" if prediction == 1 else "Negative"
        ),
        "model_probability_score": round(
            probability_score * 100, 2
        )
    }

    return result


# Pima Diabetes Prediction
def predict_pima(input_data):

    model = load_pima_model()

    return generate_prediction(
        model,
        input_data
    )


# Early Stage Diabetes Prediction
def predict_early_diabetes(input_data):

    model = load_early_diabetes_model()

    return generate_prediction(
        model,
        input_data
    )


# Testing the prediction engine
if __name__ == "__main__":

    print("\nDIABETES RISK PREDICTION ENGINE")
    print("=" * 50)

    # Sample Pima input
    pima_input = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }

    print("\nPIMA DIABETES PREDICTION")

    pima_result = predict_pima(pima_input)

    print(pima_result)

    # Sample Early Diabetes input
    early_input = {
        "Age": 40,
        "Gender": "Male",
        "Polyuria": "No",
        "Polydipsia": "Yes",
        "sudden weight loss": "No",
        "weakness": "Yes",
        "Polyphagia": "No",
        "Genital thrush": "No",
        "visual blurring": "No",
        "Itching": "No",
        "Irritability": "No",
        "delayed healing": "No",
        "partial paresis": "No",
        "muscle stiffness": "Yes",
        "Alopecia": "Yes",
        "Obesity": "Yes"
    }

    print("\nEARLY STAGE DIABETES PREDICTION")

    early_result = predict_early_diabetes(
        early_input
    )

    print(early_result)

    print("\nPrediction engine executed successfully!")
