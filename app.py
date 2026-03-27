import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("hypertension_model.pkl")
encoders = joblib.load("label_encoders.pkl")
feature_cols = joblib.load("feature_columns.pkl")

REQUIRED_FIELDS = ["age", "bmi", "total_cholesterol_mg_dl", "ldl_mg_dl",
                   "hdl_mg_dl", "creatinine_mg_dl"]

def validate_inputs(patient_data: dict):
    missing = [f for f in REQUIRED_FIELDS if patient_data.get(f) is None]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

def predict_patient_risk(patient_data: dict):
    """
    patient_data keys: age, sex (Male/Female), residence (Urban/Rural), bmi,
    family_history_hypertension (bool), diabetes (bool), smoking (bool),
    alcohol_heavy (bool), physically_active (bool), high_salt_diet (bool),
    stroke_history (bool), myocardial_infarction (bool), heart_failure (bool),
    total_cholesterol_mg_dl, ldl_mg_dl, hdl_mg_dl, creatinine_mg_dl
    """
    validate_inputs(patient_data)

    data = patient_data.copy()
    data["sex"] = encoders["sex"].transform([data["sex"]])[0]
    data["residence"] = encoders["residence"].transform([data["residence"]])[0]

    sample_df = pd.DataFrame([data], columns=feature_cols)
    prob_hypertensive = model.predict_proba(sample_df)[0][0]   # class 0 = Hypertensive

    if prob_hypertensive >= 0.70:
        status = "High Risk (Hypertensive)"
    elif prob_hypertensive >= 0.40:
        status = "Medium Risk"
    else:
        status = "Low Risk (Healthy)"

    return {
        "status": status,
        "probability": round(prob_hypertensive * 100, 2)
    }


# ── Example usage ──
if __name__ == "__main__":
    patient = {
        "age": 58,
        "sex": "Male",
        "residence": "Urban",
        "bmi": 29.5,
        "family_history_hypertension": True,
        "diabetes": False,
        "smoking": True,
        "alcohol_heavy": False,
        "physically_active": False,
        "high_salt_diet": True,
        "stroke_history": False,
        "myocardial_infarction": False,
        "heart_failure": False,
        "total_cholesterol_mg_dl": 210,
        "ldl_mg_dl": 140,
        "hdl_mg_dl": 38,
        "creatinine_mg_dl": 1.2
    }
    result = predict_patient_risk(patient)
    print(f"Status:      {result['status']}")
    print(f"Probability: {result['probability']}%")
