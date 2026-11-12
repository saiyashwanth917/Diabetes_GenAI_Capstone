import streamlit as st
import pandas as pd

from src.predict import (
    predict_pima,
    predict_early_diabetes,
    load_pima_model,
    load_early_diabetes_model,
)

from src.genai_explainer import generate_patient_explanation

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DiabetesCare | Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PROFESSIONAL UI STYLES
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --navy: #142b4a;
        --navy-deep: #0d1f38;
        --teal: #16a6a1;
        --teal-dark: #0d8581;
        --ink: #1b2b42;
        --muted: #718096;
        --line: #e7edf3;
        --surface: #ffffff;
        --canvas: #f4f7fb;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(244,247,251,0.92);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102744 0%, #17395d 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #edf5ff !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #b8cbe0 !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 7px;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.12);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 10px 0 24px 0;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        border-radius: 13px;
        background: linear-gradient(135deg, #20c5bd, #0b898d);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 22px;
        font-weight: 800;
        box-shadow: 0 8px 20px rgba(16,177,174,0.25);
    }

    .brand-name {
        color: white;
        font-family: 'Manrope', sans-serif;
        font-size: 18px;
        font-weight: 800;
        line-height: 1.15;
    }

    .brand-caption {
        color: #afc4dc;
        font-size: 11px;
        margin-top: 3px;
        letter-spacing: .7px;
        text-transform: uppercase;
    }

    .sidebar-label {
        color: #8fa9c5 !important;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin: 16px 0 9px 2px;
    }

    .main-heading {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(25px, 3vw, 34px);
        line-height: 1.2;
        font-weight: 800;
        color: var(--navy);
        margin: 2px 0 7px 0;
    }

    .sub-heading {
        color: var(--muted);
        font-size: 14px;
        margin-bottom: 22px;
    }

    .eyebrow {
        color: var(--teal-dark);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 800;
        font-size: 10px;
        margin-bottom: 9px;
    }

    .hero {
        background: linear-gradient(115deg, #102744 0%, #1b496b 62%, #167d83 100%);
        border-radius: 22px;
        padding: 30px 32px;
        color: white;
        margin: 4px 0 22px 0;
        box-shadow: 0 14px 34px rgba(19,53,83,.13);
    }

    .hero h1 {
        color: white;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(25px, 3vw, 36px);
        font-weight: 800;
        line-height: 1.18;
        margin: 0 0 10px 0;
    }

    .hero p {
        color: #d5e5f2;
        font-size: 14px;
        max-width: 650px;
        margin: 0;
        line-height: 1.7;
    }

    .hero-tag {
        display: inline-block;
        background: rgba(255,255,255,.12);
        border: 1px solid rgba(255,255,255,.19);
        color: #e8ffff;
        padding: 6px 11px;
        border-radius: 30px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 16px;
    }

    .stat-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 19px;
        min-height: 112px;
        box-shadow: 0 4px 16px rgba(25,48,75,.035);
    }

    .stat-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .stat-value {
        color: var(--navy);
        font-family: 'Manrope', sans-serif;
        font-size: 25px;
        font-weight: 800;
        line-height: 1.1;
    }

    .stat-note {
        color: #8795a8;
        font-size: 11px;
        margin-top: 7px;
    }

    .section-title {
        color: var(--navy);
        font-family: 'Manrope', sans-serif;
        font-size: 18px;
        font-weight: 800;
        margin: 22px 0 4px 0;
    }

    .section-note {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 14px;
    }

    .feature-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 20px;
        min-height: 180px;
        box-shadow: 0 5px 18px rgba(25,48,75,.035);
    }

    .feature-icon {
        width: 43px;
        height: 43px;
        border-radius: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #e8f8f7;
        font-size: 21px;
        margin-bottom: 14px;
    }

    .feature-card h3 {
        color: var(--navy);
        font-size: 15px;
        font-weight: 800;
        margin: 0 0 7px 0;
    }

    .feature-card p {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.6;
        margin: 0;
    }

    .panel {
        background: white;
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 20px 22px;
        margin: 12px 0;
        box-shadow: 0 4px 16px rgba(25,48,75,.025);
    }

    .result-positive {
        background: #fff5f3;
        border: 1px solid #ffd9d2;
        border-left: 5px solid #e36b58;
        border-radius: 15px;
        padding: 20px 22px;
    }

    .result-negative {
        background: #effaf6;
        border: 1px solid #ccefe1;
        border-left: 5px solid #22a879;
        border-radius: 15px;
        padding: 20px 22px;
    }

    .result-kicker {
        color: #758397;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .result-title {
        color: var(--navy);
        font-family: 'Manrope', sans-serif;
        font-size: 25px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .result-copy {
        color: #64748b;
        font-size: 12px;
        line-height: 1.6;
    }

    div[data-testid="stForm"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 22px 22px 16px 22px;
        box-shadow: 0 4px 16px rgba(25,48,75,.025);
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        color: #31445c;
        font-weight: 700;
        font-size: 12px;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 10px;
        border-color: #dce5ee;
        background: #fbfcfe;
    }

    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(100deg, #117e87, #19aaa4);
        color: white;
        border: 0;
        border-radius: 10px;
        padding: 11px 18px;
        font-weight: 800;
        min-height: 44px;
        transition: all .2s ease;
    }

    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        color: white;
        border: 0;
        background: linear-gradient(100deg, #0d6973, #128e8b);
        box-shadow: 0 7px 18px rgba(17,126,135,.2);
        transform: translateY(-1px);
    }

    /* Force readable metric text even when Streamlit is running in dark mode */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e1e9f1 !important;
        padding: 16px 18px !important;
        border-radius: 13px !important;
        color: #142b4a !important;
    }

    [data-testid="stMetric"] *,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div,
    [data-testid="stMetricDelta"] {
        color: #142b4a !important;
        opacity: 1 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 800 !important;
    }

    /* Keep progress indicator visible and on-brand */
    [data-testid="stProgress"] > div > div {
        background-color: #e3edf3 !important;
        border-radius: 20px !important;
    }
    [data-testid="stProgress"] [role="progressbar"] > div {
        background-color: #16a6a1 !important;
        border-radius: 20px !important;
    }

    /* Accessible warning/disclaimer: dark text on a soft, readable background */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        background: #fff8e8 !important;
        border: 1px solid #f2d99b !important;
        color: #624b16 !important;
    }

    [data-testid="stAlert"] *,
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        color: #624b16 !important;
        opacity: 1 !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    .footer {
        color: #8a98a9;
        text-align: center;
        font-size: 11px;
        padding: 24px 0 10px 0;
    }

    @media (max-width: 700px) {
        .hero { padding: 23px 20px; border-radius: 16px; }
        .panel { padding: 16px; }
        div[data-testid="stForm"] { padding: 15px 13px; }
        .stat-card { padding: 15px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS AND HELPERS
# ============================================================

PIMA_FIELDS = {
    "Pregnancies": {
        "label": "Number of Pregnancies",
        "unit": "Count",
        "help": "Enter the number of pregnancies.",
    },
    "Glucose": {
        "label": "Glucose",
        "unit": "mg/dL",
        "help": "Pima uses 2-hour plasma glucose. Below 140 mg/dL is generally normal for a non-pregnant adult.",
    },
    "BloodPressure": {
        "label": "Diastolic Blood Pressure",
        "unit": "mmHg",
        "help": "Below 80 mmHg is generally considered normal diastolic blood pressure.",
    },
    "SkinThickness": {
        "label": "Skin Thickness",
        "unit": "mm",
        "help": "Dataset measurement; no universal clinical reference range.",
    },
    "Insulin": {
        "label": "Insulin",
        "unit": "µU/mL",
        "help": "No single standard clinical reference range applies to this dataset measurement.",
    },
    "BMI": {
        "label": "Body Mass Index (BMI)",
        "unit": "kg/m²",
        "help": "18.5–24.9 is the general adult healthy BMI range.",
    },
    "DiabetesPedigreeFunction": {
        "label": "Diabetes Pedigree Function",
        "unit": "Score",
        "help": "Dataset-specific family-history score; no established clinical normal range.",
    },
    "Age": {
        "label": "Age",
        "unit": "Years",
        "help": "Enter age in years.",
    },
}


def get_reference_status(feature, value):
    if feature == "Glucose":
        return ("Within reference", "normal") if value < 140 else ("Above reference", "high")
    if feature == "BloodPressure":
        return ("Within reference", "normal") if value < 80 else ("Above reference", "high")
    if feature == "BMI":
        if 18.5 <= value <= 24.9:
            return "Within reference", "normal"
        return ("Below reference", "low") if value < 18.5 else ("Above reference", "high")
    return "No established reference range", "unknown"


def page_heading(eyebrow, title, description):
    st.markdown(
        f"""
        <div class="eyebrow">{eyebrow}</div>
        <div class="main-heading">{title}</div>
        <div class="sub-heading">{description}</div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title, note=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


def show_prediction_result(result, patient_name, assessment_name, summary, patient_inputs, state_key):
    prediction = result["prediction"]
    probability = result["model_probability_score"]

    st.markdown("---")
    section_heading("Assessment result", "Review the model output and the information entered.")

    result_class = "result-positive" if prediction == 1 else "result-negative"
    result_title = "Positive model screening result" if prediction == 1 else "Negative model screening result"
    result_copy = (
        "The model classified this input as positive. This is a screening result, not a diagnosis."
        if prediction == 1
        else "The model classified this input as negative. A negative result does not rule out diabetes."
    )

    st.markdown(
        f"""
        <div class="{result_class}">
            <div class="result-kicker">{assessment_name}</div>
            <div class="result-title">{result_title}</div>
            <div class="result-copy">Patient: {patient_name}</div>
            <div class="result-copy">{result_copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Model-estimated positive-class probability", f"{probability}%")
        st.progress(max(0.0, min(float(probability) / 100, 1.0)))
    with col2:
        st.markdown(
            """
            <div class="panel">
                <div class="result-kicker">Important interpretation</div>
                <div style="color:#31445c;font-size:13px;line-height:1.65;">
                This percentage is the model's estimated positive-class probability.
                It is not model accuracy and is not a confirmed probability of disease.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_heading("Patient input summary")
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    st.warning(
        "This application is for educational screening support only. It does not provide "
        "a medical diagnosis. Please consult a qualified healthcare professional for "
        "clinical evaluation and testing."
    )

    # --------------------------------------------------------
    # GEMINI AI EXPLANATION
    # --------------------------------------------------------
    section_heading(
        "AI-powered explanation",
        "A patient-friendly explanation of the model output."
    )

    explanation_key = f"{state_key}_ai_explanation"
    error_key = f"{state_key}_ai_error"

    if st.button(
        "Generate AI explanation",
        key=f"generate_ai_{state_key}",
        use_container_width=True,
    ):
        try:
            # The explainer expects a human-readable result label.
            explanation_result = dict(result)
            explanation_result.setdefault(
                "result",
                "Positive" if int(result["prediction"]) == 1 else "Negative"
            )

            with st.spinner("Gemini is preparing the patient-friendly explanation..."):
                st.session_state[explanation_key] = generate_patient_explanation(
                    dataset_name=assessment_name,
                    patient_inputs=patient_inputs,
                    prediction_result=explanation_result,
                )

            st.session_state.pop(error_key, None)

        except Exception as error:
            st.session_state.pop(explanation_key, None)
            st.session_state[error_key] = str(error)

    if st.session_state.get(explanation_key):
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(st.session_state[explanation_key])
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get(error_key):
        st.error(
            "The ML prediction is complete, but Gemini could not generate "
            "the explanation. Your prediction result is still available. "
            f"Details: {st.session_state[error_key]}"
        )


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def get_models():
    pima = load_pima_model()
    early = load_early_diabetes_model()
    return pima, early


try:
    pima_model, early_model = get_models()
    pima_features = list(pima_model.feature_names_in_)
    early_features = list(early_model.feature_names_in_)
    models_ready = True
except Exception as error:
    models_ready = False
    model_error = str(error)
    pima_features = []
    early_features = []

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">✚</div>
            <div>
                <div class="brand-name">DiabetesCare</div>
                <div class="brand-caption">AI risk assessment</div>
            </div>
        </div>
        <div class="sidebar-label">WORKSPACE</div>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        "Dashboard",
        "Pima Diabetes",
        "Early Stage Diabetes",
        "About Project",
    ]
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Dashboard"

    selected_page = st.radio(
        "Navigate",
        pages,
        index=pages.index(st.session_state.active_page),
        label_visibility="collapsed",
    )
    st.session_state.active_page = selected_page

    st.markdown("---")
    st.markdown(
        """
        <div class="sidebar-label">SYSTEM STATUS</div>
        <div style="font-size:12px;line-height:1.8;">
            <span style="color:#58d6bb;">●</span> Application interface active<br>
            <span style="color:#58d6bb;">●</span> ML assessment modules
        </div>
        <div style="margin-top:24px;color:#8fa9c5;font-size:10px;line-height:1.6;">
            CAPSTONE PROJECT<br>
            Machine Learning + Generative AI
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MODEL ERROR HANDLING
# ============================================================

if not models_ready:
    st.error(f"Unable to load trained models: {model_error}")
    st.info(
        "Check that your trained model files are present and that the functions in "
        "`src/predict.py` are working correctly."
    )
    st.stop()

# ============================================================
# DASHBOARD
# ============================================================

if selected_page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-tag">✦ INTELLIGENT HEALTH SCREENING</div>
            <h1>Understand diabetes risk.<br>Make informed next steps.</h1>
            <p>
                A machine-learning powered screening interface that evaluates health
                measurements and reported symptoms using two trained prediction models.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="stat-card"><div class="stat-label">ASSESSMENT MODULES</div>'
            '<div class="stat-value">02</div><div class="stat-note">Pima and Early Stage Diabetes</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="stat-card"><div class="stat-label">PREDICTION ENGINE</div>'
            '<div class="stat-value">ML</div><div class="stat-note">Uses your trained classification models</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="stat-card"><div class="stat-label">EXPLANATION LAYER</div>'
            '<div class="stat-value">GenAI</div><div class="stat-note">Gemini explanation integration</div></div>',
            unsafe_allow_html=True,
        )

    section_heading("Choose an assessment", "Select a model to begin a new screening assessment.")

    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🧬</div>
                <h3>Pima Diabetes Assessment</h3>
                <p>Evaluate eight numerical health measurements, including glucose,
                BMI, blood pressure, insulin and age.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Pima assessment →", use_container_width=True, key="go_pima"):
            st.session_state.active_page = "Pima Diabetes"
            st.rerun()

    with right:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🩺</div>
                <h3>Early Stage Diabetes Assessment</h3>
                <p>Evaluate age, gender and reported symptoms using the early-stage
                diabetes classification model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open early-stage assessment →", use_container_width=True, key="go_early"):
            st.session_state.active_page = "Early Stage Diabetes"
            st.rerun()

    section_heading("How it works", "A simple workflow from patient information to model output.")
    a, b, c = st.columns(3)
    for col, number, title, description in [
        (a, "01", "Enter information", "Provide the requested patient measurements or symptoms."),
        (b, "02", "Run prediction", "The selected trained model processes the supplied inputs."),
        (c, "03", "Review result", "View the model classification and positive-class probability."),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="panel">
                    <div style="color:#16a6a1;font-size:11px;font-weight:800;letter-spacing:1px;">STEP {number}</div>
                    <div style="color:#142b4a;font-size:15px;font-weight:800;margin:8px 0;">{title}</div>
                    <div style="color:#718096;font-size:12px;line-height:1.6;">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "Educational screening support only — not a substitute for professional medical advice."
    )

# ============================================================
# PIMA DIABETES ASSESSMENT
# ============================================================

elif selected_page == "Pima Diabetes":
    page_heading(
        "ASSESSMENT  /  MODEL 01",
        "Pima Diabetes Assessment",
        "Enter the patient's details and eight numerical measurements to run the Pima model.",
    )

    st.info(
        "Important: In the Pima dataset, zero values for glucose, blood pressure, skin "
        "thickness, insulin and BMI may represent missing measurements. Enter actual "
        "measured values instead of using zero as a substitute."
    )

    with st.form("pima_form", clear_on_submit=False):
        section_heading("Patient information", "Basic details for this assessment.")
        patient_name = st.text_input(
            "Patient name",
            placeholder="Enter patient's full name",
            key="pima_patient_name",
        )

        st.markdown("---")
        section_heading("Medical measurements", "Enter the values from the patient's measurements.")

        pima_input = {}
        col1, col2 = st.columns(2)

        for index, feature in enumerate(pima_features):
            details = PIMA_FIELDS.get(
                feature,
                {"label": feature, "unit": "", "help": "Enter the measured value."},
            )
            current_col = col1 if index % 2 == 0 else col2

            with current_col:
                if feature == "Pregnancies":
                    value = st.number_input(
                        details["label"],
                        min_value=0,
                        value=0,
                        step=1,
                        help=details["help"],
                        key="pima_Pregnancies",
                    )
                elif feature == "Age":
                    value = st.number_input(
                        "Age (years)",
                        min_value=1,
                        max_value=120,
                        value=30,
                        step=1,
                        help=details["help"],
                        key="pima_Age",
                    )
                else:
                    value = st.number_input(
                        f'{details["label"]} ({details["unit"]})',
                        min_value=0.0,
                        value=0.0,
                        step=0.1,
                        help=details["help"],
                        key=f"pima_{feature}",
                    )

                pima_input[feature] = value

                if feature in ["Glucose", "BloodPressure", "BMI"]:
                    status, category = get_reference_status(feature, value)
                    if category == "normal":
                        st.caption(f"🟢 {status}")
                    elif category in ["high", "low"]:
                        st.caption(f"🟠 {status}")

        submitted = st.form_submit_button(
            "Run diabetes risk assessment",
            use_container_width=True,
        )

    if submitted:
        if not patient_name.strip():
            st.error("Please enter the patient's name.")
        else:
            missing = [
                name
                for name in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
                if name in pima_input and pima_input[name] == 0
            ]
            if missing:
                st.error(
                    "Please enter actual measurements instead of zero for: "
                    + ", ".join(missing)
                )
            else:
                try:
                    with st.spinner("Analyzing measurements with the Pima model..."):
                        result = predict_pima(pima_input)

                    summary = []
                    for feature, value in pima_input.items():
                        details = PIMA_FIELDS.get(feature, {})
                        status, _ = get_reference_status(feature, value)
                        summary.append(
                            {
                                "Measurement": details.get("label", feature),
                                "Value": value,
                                "Unit": details.get("unit", ""),
                                "Reference assessment": status,
                            }
                        )

                    st.session_state["pima_assessment"] = {
                        "result": result,
                        "patient_name": patient_name.strip(),
                        "assessment_name": "Pima Diabetes Model",
                        "summary": summary,
                        "patient_inputs": dict(pima_input),
                    }
                    st.session_state.pop("pima_assessment_ai_explanation", None)
                    st.session_state.pop("pima_assessment_ai_error", None)
                except Exception as error:
                    st.error(f"Prediction error: {error}")

    if st.session_state.get("pima_assessment"):
        saved = st.session_state["pima_assessment"]
        show_prediction_result(
            saved["result"],
            saved["patient_name"],
            saved["assessment_name"],
            saved["summary"],
            saved["patient_inputs"],
            "pima_assessment",
        )

# ============================================================
# EARLY STAGE DIABETES ASSESSMENT
# ============================================================

elif selected_page == "Early Stage Diabetes":
    page_heading(
        "ASSESSMENT  /  MODEL 02",
        "Early Stage Diabetes Assessment",
        "Enter the patient's details and reported symptoms to run the early-stage model.",
    )

    st.info(
        "This assessment uses categorical symptom values such as Yes and No. "
        "These are model inputs, not numeric medical reference ranges."
    )

    with st.form("early_diabetes_form", clear_on_submit=False):
        section_heading("Patient information", "Basic details for this assessment.")
        patient_name = st.text_input(
            "Patient name",
            placeholder="Enter patient's full name",
            key="early_patient_name",
        )

        st.markdown("---")
        section_heading("Medical information", "Select the values that best match the patient's reported information.")

        early_input = {}
        col1, col2 = st.columns(2)

        for index, feature in enumerate(early_features):
            current_col = col1 if index % 2 == 0 else col2

            with current_col:
                if feature == "Age":
                    value = st.number_input(
                        "Age (years)",
                        min_value=1,
                        max_value=120,
                        value=30,
                        step=1,
                        key="early_age",
                    )
                elif feature == "Gender":
                    value = st.selectbox(
                        "Gender",
                        ["Male", "Female"],
                        key="early_gender",
                    )
                else:
                    value = st.selectbox(
                        feature.replace("_", " "),
                        ["No", "Yes"],
                        key=f"early_{feature}",
                    )

                early_input[feature] = value

        submitted = st.form_submit_button(
            "Run early-stage risk assessment",
            use_container_width=True,
        )

    if submitted:
        if not patient_name.strip():
            st.error("Please enter the patient's name.")
        else:
            try:
                with st.spinner("Analyzing patient information with the early-stage model..."):
                    result = predict_early_diabetes(early_input)

                summary = [
                    {"Input feature": feature.replace("_", " "), "Entered value": value}
                    for feature, value in early_input.items()
                ]

                st.session_state["early_assessment"] = {
                    "result": result,
                    "patient_name": patient_name.strip(),
                    "assessment_name": "Early Stage Diabetes Model",
                    "summary": summary,
                    "patient_inputs": dict(early_input),
                }
                st.session_state.pop("early_assessment_ai_explanation", None)
                st.session_state.pop("early_assessment_ai_error", None)
            except Exception as error:
                st.error(f"Prediction error: {error}")

    if st.session_state.get("early_assessment"):
        saved = st.session_state["early_assessment"]
        show_prediction_result(
            saved["result"],
            saved["patient_name"],
            saved["assessment_name"],
            saved["summary"],
            saved["patient_inputs"],
            "early_assessment",
        )

# ============================================================
# ABOUT PROJECT
# ============================================================

elif selected_page == "About Project":
    page_heading(
        "PROJECT INFORMATION",
        "About DiabetesCare",
        "An academic capstone project combining machine learning and generative AI concepts.",
    )

    st.markdown(
        """
        <div class="panel">
            <div class="eyebrow">PROJECT OVERVIEW</div>
            <div style="color:#142b4a;font-family:Manrope,sans-serif;font-size:20px;font-weight:800;margin-bottom:9px;">
                Diabetes Risk Prediction using Machine Learning and GenAI
            </div>
            <div style="color:#718096;font-size:13px;line-height:1.8;">
                The application provides two assessment pathways: a Pima diabetes
                classification model based on numerical measurements and an early-stage
                diabetes model based on demographic information and reported symptoms.
                A generative AI explanation layer is intended to make model outputs easier
                to understand when that integration is available.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_heading("Application modules")
    st.markdown(
        """
        - **Pima Diabetes:** uses numerical patient measurements.
        - **Early Stage Diabetes:** uses age, gender and categorical symptom inputs.
        - **Prediction engine:** calls the trained model functions in `src/predict.py`.
        - **GenAI explanation:** intended to provide plain-language explanations of model results.
        """
    )

    st.warning(
        "Medical disclaimer: This capstone application is an educational prototype. "
        "Its outputs must not be used to diagnose, treat or rule out diabetes. "
        "Clinical decisions require evaluation by a qualified healthcare professional."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        DiabetesCare · Diabetes Risk Prediction Capstone Project<br>
        Machine Learning · Generative AI · Educational Screening Support
    </div>
    """,
    unsafe_allow_html=True,
)
