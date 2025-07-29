import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Robust Path to Model ---
# This creates an absolute path to the model file, making it work reliably.
try:
    # Get the absolute path to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go one level up to the project root (from 'ui' to 'Heart_Disease_Project')
    project_root = os.path.dirname(script_dir)
    # Construct the full path to the model file
    model_path = os.path.join(project_root, 'models', 'final_model.pkl')
except NameError:
    # Fallback for environments where __file__ is not defined (like some notebooks)
    model_path = os.path.join('models', 'final_model.pkl')


# --- Load The Model ---
# We cache the model loading so it doesn't re-load on every interaction.
@st.cache_resource
def load_model(path):
    """Loads the trained machine learning model from a file."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {path}. Please ensure the model is trained and saved correctly in the 'models' directory.")
        return None

model = load_model(model_path)


# --- UI Components ---
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This application uses a machine learning model to predict the likelihood of heart disease
based on user-provided health data. Please fill in the details in the sidebar to get a prediction.

**Disclaimer:** This prediction is not a substitute for professional medical advice.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Patient Health Information")
st.sidebar.markdown("Please provide the following details:")

# Function to collect user inputs
def get_user_input():
    """Creates sidebar widgets and collects user input."""
    # The features must be in the exact order the model was trained on.
    # The error message indicates the model expects: ['sex', 'cp', 'fbs', 'restecg', 'exang', 'oldpeak', 'slope', 'ca']

    sex_options = {0: "Female", 1: "Male"}
    sex = st.sidebar.selectbox("Sex", options=list(sex_options.keys()), format_func=lambda x: sex_options[x])

    cp_options = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
    cp = st.sidebar.selectbox("Chest Pain Type (CP)", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])

    fbs_options = {0: "False (< 120 mg/dl)", 1: "True (> 120 mg/dl)"}
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=list(fbs_options.keys()), format_func=lambda x: fbs_options[x])

    restecg_options = {0: "Normal", 1: "ST-T wave abnormality", 2: "Probable or definite left ventricular hypertrophy"}
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results (restecg)", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])

    exang_options = {0: "No", 1: "Yes"}
    exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", options=list(exang_options.keys()), format_func=lambda x: exang_options[x])

    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise (oldpeak)", 0.0, 6.2, 1.0)

    slope_options = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
    slope = st.sidebar.selectbox("Slope of the peak exercise ST segment", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])

    ca = st.sidebar.slider("Number of Major Vessels Colored by Flourosopy (ca)", 0, 4, 0)

    # Create a dictionary of the inputs in the correct order
    input_data = {
        'sex': sex,
        'cp': cp,
        'fbs': fbs,
        'restecg': restecg,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca
    }

    # Convert to a DataFrame
    features = pd.DataFrame(input_data, index=[0])
    return features

# Get user input
user_input_df = get_user_input()

# --- Display User Input ---
st.subheader("Your Input Summary")
st.write(user_input_df)

# --- Prediction Logic ---
if st.button("Get Prediction", key="predict_button"):
    if model is not None:
        # Make prediction
        prediction = model.predict(user_input_df)
        prediction_proba = model.predict_proba(user_input_df)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("High Risk of Heart Disease Detected", icon="🚨")
            st.write(f"**Confidence:** {prediction_proba[0][1]*100:.2f}%")
            st.markdown("It is highly recommended to consult with a healthcare professional for a thorough evaluation.")
        else:
            st.success("Low Risk of Heart Disease Detected", icon="✅")
            st.write(f"**Confidence:** {prediction_proba[0][0]*100:.2f}%")
            st.markdown("Continue to maintain a healthy lifestyle. Regular check-ups are always a good practice.")
    else:
        st.warning("Model is not loaded. Cannot make a prediction.")
