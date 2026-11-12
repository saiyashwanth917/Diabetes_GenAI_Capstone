import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError


# Load the project's .env file (project root is the parent of src/).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")


def get_gemini_client():
    """Create a Gemini client with a finite request timeout."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY=your_key to the .env "
            "file in the main Diabetes_GenAI_Capstone folder, then restart Streamlit."
        )

    # Timeout is in milliseconds. This prevents the Streamlit spinner from
    # waiting indefinitely if the API connection stalls.
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30000),
    )


def generate_patient_explanation(dataset_name, patient_inputs, prediction_result):
    """Generate a safe, patient-friendly explanation of an ML screening result."""
    client = get_gemini_client()

    prediction = prediction_result.get("result", "Unknown")
    probability = prediction_result.get("model_probability_score", "Not available")

    measurements = "\n".join(
        f"{feature}: {value}" for feature, value in patient_inputs.items()
    )

    prompt = f"""
You are a healthcare education assistant supporting a machine-learning diabetes
screening project. Explain this result in simple, supportive English.

DATASET: {dataset_name}
PATIENT MEASUREMENTS:
{measurements}

MACHINE-LEARNING SCREENING RESULT: {prediction}
MODEL-ESTIMATED POSITIVE-CLASS PROBABILITY: {probability}%

Include these headings:
### 1. Prediction Explanation
### 2. Understanding the Model Probability
### 3. Important Measurements
### 4. Recommended Next Steps
### 5. Medical Disclaimer

Explain that the result is a model screening output, not a diagnosis. Explain that
model probability is not model accuracy or diagnostic certainty. Discuss only the
measurements supplied; do not invent values, reference ranges, causes, or feature
importance. Do not prescribe medication or treatment. Recommend discussing
appropriate clinical testing with a qualified healthcare professional.
"""

    # Try the configured model first. If Google reports temporary overload (503),
    # switch to a second supported Flash model instead of leaving the UI waiting.
    # Each request has a 30-second timeout in get_gemini_client().
    models_to_try = [PRIMARY_MODEL]
    if FALLBACK_MODEL and FALLBACK_MODEL != PRIMARY_MODEL:
        models_to_try.append(FALLBACK_MODEL)

    last_error = None

    for model_index, model_name in enumerate(models_to_try):
        for attempt in range(1, 3):
            try:
                print(
                    f"Connecting to Gemini ({model_name}), "
                    f"attempt {attempt}/2..."
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )

                explanation = getattr(response, "text", None)
                if explanation and explanation.strip():
                    return explanation.strip()

                raise RuntimeError(
                    f"Gemini ({model_name}) returned an empty response."
                )

            except APIError as error:
                last_error = error
                status_code = getattr(error, "code", None)

                if status_code == 429:
                    raise RuntimeError(
                        "Gemini API quota/rate limit reached (HTTP 429). "
                        "Check the API project's rate limits in Google AI Studio."
                    ) from error

                if status_code in (500, 502, 503, 504):
                    if attempt < 2:
                        wait_seconds = 2 * attempt
                        print(
                            f"Gemini temporarily unavailable (HTTP {status_code}); "
                            f"retrying in {wait_seconds} seconds..."
                        )
                        time.sleep(wait_seconds)
                        continue

                    # Move to the fallback model after the primary model's retries.
                    print(f"{model_name} is still unavailable; checking fallback model.")
                    break

                raise RuntimeError(
                    f"Gemini API request failed for {model_name} "
                    f"(HTTP {status_code}): {error}"
                ) from error

            except (TimeoutError, OSError) as error:
                last_error = error
                if attempt < 2:
                    time.sleep(2)
                    continue
                print(f"{model_name} timed out; checking fallback model.")
                break

            except Exception as error:
                raise RuntimeError(
                    f"Unable to generate the AI explanation with {model_name}: {error}"
                ) from error

    raise RuntimeError(
        "Gemini could not generate the explanation using the primary or fallback "
        f"model. Last error: {last_error}. Your ML prediction is still available; "
        "please try the AI explanation again later."
    )


if __name__ == "__main__":
    sample_inputs = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 120,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }
    sample_prediction = {
        "prediction": 1,
        "result": "Positive",
        "model_probability_score": 75.0,
    }

    try:
        print(generate_patient_explanation("Pima Diabetes", sample_inputs, sample_prediction))
    except Exception as error:
        print(f"\nGENAI ERROR: {error}")
