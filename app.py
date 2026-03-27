import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("best_hypertension_model.pkl", "rb"))

st.set_page_config(page_title="Hypertension Predictor", layout="centered")

st.title("🩺 Hypertension Risk Predictor")
st.write("Fill in your health details below:")

# --- INPUTS ---
age = st.slider("Age", 18, 100, 30)
bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
cholesterol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 180)

sex = st.selectbox("Sex", ["Male", "Female"])
smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
diabetes = st.selectbox("Do you have diabetes?", ["No", "Yes"])
family_history = st.selectbox("Family History of Hypertension?", ["No", "Yes"])
alcohol = st.selectbox("Heavy Alcohol Consumption?", ["No", "Yes"])
physical_activity = st.selectbox("Physically Active?", ["Yes", "No"])
high_salt = st.selectbox("High Salt Diet?", ["No", "Yes"])

# --- ENCODING ---
sex = 1 if sex == "Male" else 0
smoking = 1 if smoking == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
family_history = 1 if family_history == "Yes" else 0
alcohol = 1 if alcohol == "Yes" else 0
physical_activity = 1 if physical_activity == "Yes" else 0
high_salt = 1 if high_salt == "Yes" else 0

# --- DATAFRAME ---
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "family_history_hypertension": [family_history],
    "diabetes": [diabetes],
    "smoking": [smoking],
    "alcohol_heavy": [alcohol],
    "physically_active": [physical_activity],
    "high_salt_diet": [high_salt],
    "total_cholesterol_mg_dl": [cholesterol]
})

# --- PREDICT ---
if st.button("Predict Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result:")

    if prediction == 1:
        st.error(f"⚠️ High Risk ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk ({probability*100:.2f}%)")