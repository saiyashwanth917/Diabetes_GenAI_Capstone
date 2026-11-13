# 🩺 DiabetesCare – Diabetes Risk Prediction Using Machine Learning & Generative AI

<p align="center">
  <strong>An Intelligent Diabetes Risk Assessment and Explanation System</strong>
</p>

<p align="center">
  <a href="https://diabetesgenaicapstone-tdtwhdoxu5o7cty3fdkgov.streamlit.app/">
    🚀 **OPEN LIVE APPLICATION**
  </a>
</p>

<p align="center">
  Built with Python · Machine Learning · Streamlit · Google Gemini
</p>

---

## 📌 Project Overview

**DiabetesCare** is an interactive web-based application that combines Machine Learning (ML) and Generative Artificial Intelligence (GenAI) to provide diabetes risk screening and understandable explanations of prediction results.

The application integrates two trained machine learning models to assess diabetes risk using different datasets and input features. It also uses Google's Gemini API to generate patient-friendly explanations of prediction results.

The primary goal is to make machine learning predictions easier to understand through an interactive and user-friendly interface.

## 🌐 Live Application

**Try the application here:**

👉 https://diabetesgenaicapstone-tdtwhdoxu5o7cty3fdkgov.streamlit.app/

The application is hosted on Streamlit Community Cloud.

## ✨ Key Features

### 1. Interactive Dashboard
- Professional and user-friendly interface.
- Centralized dashboard for accessing assessment modules.
- Simple navigation between application sections.

### 2. Pima Diabetes Prediction
- Uses the Pima Indians Diabetes dataset.
- Accepts numerical medical measurements.
- Processes patient inputs using a trained machine learning pipeline.
- Displays the prediction result and model probability score.

### 3. Early Stage Diabetes Prediction
- Uses an early-stage diabetes dataset.
- Evaluates demographic information and reported symptoms.
- Supports interactive patient input.
- Displays the predicted screening result and probability score.

### 4. Generative AI Explanation Using Google Gemini
- Integrates Google's Gemini API.
- Converts machine learning results into understandable explanations.
- Explains the prediction and model probability.
- Highlights relevant patient measurements.
- Provides general recommendations for appropriate medical follow-up.

### 5. Prediction Results
- Displays prediction outcomes.
- Shows model-estimated probability scores.
- Presents patient input summaries.
- Provides AI-generated explanations when the Gemini service is available.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning and preprocessing |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web application |
| Google Gemini API | AI-generated explanations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Git & GitHub | Version control and project hosting |

## 🧠 System Architecture

The application follows this general workflow:

1. **Patient Input:** Users enter medical measurements or select reported symptoms.
2. **Data Preprocessing:** Input data is prepared using the corresponding preprocessing pipeline.
3. **Machine Learning Prediction:** The selected trained model generates a screening result.
4. **Probability Calculation:** The model provides an estimated positive-class probability where supported.
5. **GenAI Integration:** Prediction details are passed to Google Gemini for explanation.
6. **Result Presentation:** The application displays the prediction, probability and generated explanation.

## 📂 Project Structure

```text
Diabetes_GenAI_Capstone/
│
├── app.py
│
├── src/
│   ├── predict.py
│   └── genai_explainer.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── Trained model artifacts
│
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

*Note: The structure above illustrates the main project components. Refer to the repository for the actual model artifact filenames and additional files.*

## ⚙️ Installation and Setup

Follow these steps to run the application locally.

### Prerequisites

- Python 3.11 or compatible Python environment
- Git
- Google Gemini API key

### Step 1: Clone the Repository

```bash
git clone https://github.com/saiyashwanth917/Diabetes_GenAI_Capstone.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Diabetes_GenAI_Capstone
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure Gemini API

Create a `.env` file in the project root directory.

Add your Gemini API key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

Obtain your API key through Google AI Studio:

https://aistudio.google.com/apikey

**Important:** Never upload your actual API key or `.env` file to GitHub.

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## ☁️ Deployment

The project is deployed using **Streamlit Community Cloud**.

Deployment configuration:

- Repository: `saiyashwanth917/Diabetes_GenAI_Capstone`
- Branch: `master`
- Main application file: `app.py`
- Application framework: Streamlit

The Gemini API key is configured securely through Streamlit Cloud Secrets.

## 🔐 Environment Variables

The application requires the following environment variable for Gemini integration:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API authentication key |

Keep API credentials private and use environment variables or Streamlit Secrets.

## 🎯 Project Objectives

- Develop a machine learning-based diabetes risk screening system.
- Integrate multiple trained prediction models.
- Explore data preprocessing and machine learning techniques.
- Integrate Generative AI for understandable prediction explanations.
- Develop an interactive application using Streamlit.
- Deploy the application through a cloud platform.

## 🔮 Future Enhancements

- Improve model performance through systematic experimentation.
- Add more machine learning algorithms for comparison.
- Enhance prediction explainability.
- Introduce downloadable assessment reports.
- Improve model monitoring and validation.
- Expand the application with additional healthcare screening modules.

## ⚠️ Medical Disclaimer

This application is an academic capstone project developed for educational and demonstration purposes.

The predictions generated by the machine learning models are screening estimates and **must not be considered medical diagnoses**.

The probability score represents a model-estimated probability, not the model's accuracy or a guarantee of an individual's medical condition.

Users should consult qualified healthcare professionals for medical evaluation, diagnosis and treatment decisions.

## 👨‍💻 Developer

**Sai Yashwanth**

GitHub: https://github.com/saiyashwanth917

**Project:** Diabetes GenAI Capstone

---

<p align="center">
  <strong>DiabetesCare – Combining Machine Learning and Generative AI for Understandable Health Screening</strong>
</p>

<p align="center">
  ⭐ If you find this project interesting, consider starring the repository!
</p>
