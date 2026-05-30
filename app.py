  
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="centered"
)

# Load model & scaler
@st.cache_resource
def load_model():
    model  = joblib.load('model_xgb_tuned.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Header
st.title("🏥 Hospital Readmission Risk Predictor")
st.markdown("Predict whether a diabetic patient will be **readmitted within 30 days** of discharge.")
st.divider()

# Input form
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.selectbox("Age Group",
          options=[5,15,25,35,45,55,65,75,85,95],
          format_func=lambda x: {
              5:'0-10', 15:'10-20', 25:'20-30', 35:'30-40',
              45:'40-50', 55:'50-60', 65:'60-70', 75:'70-80',
              85:'80-90', 95:'90-100'}[x])

    time_in_hospital     = st.slider("Days in Hospital", 1, 14, 3)
    num_lab_procedures   = st.slider("Number of Lab Procedures", 1, 132, 40)
    num_medications      = st.slider("Number of Medications", 1, 81, 15)
    number_inpatient     = st.slider("Inpatient Visits (past year)", 0, 21, 0)

with col2:
    number_emergency     = st.slider("Emergency Visits (past year)", 0, 76, 0)
    number_diagnoses     = st.slider("Number of Diagnoses", 1, 16, 5)
    total_encounters     = st.slider("Total Past Encounters", 1, 40, 1)
    max_inpatient_history     = st.slider("Max Inpatient Visits (historical)", 0, 21, 0)
    mean_medications_history  = st.slider("Avg Medications (historical)", 1.0, 81.0, 15.0)

st.divider()
st.subheader("Medical Details")

col3, col4 = st.columns(2)

with col3:
    ever_readmitted = st.selectbox("Ever Readmitted Before?", ['No', 'Yes'])
    ever_readmitted_val = 1 if ever_readmitted == 'Yes' else 0

    metformin = st.selectbox("Metformin", ['No', 'Steady', 'Up', 'Down'])
    metformin_map = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}

    race = st.selectbox("Race", ['Other', 'Caucasian'])
    race_2_val = 1 if race == 'Caucasian' else 0

with col4:
    discharge_disposition = st.selectbox("Discharge Disposition", [
        'Discharged to home',
        'Transferred to another facility',
        'Left against medical advice',
        'Expired / Hospice'
    ])
    discharge_map = {
        'Discharged to home'             : 1,
        'Transferred to another facility': 2,
        'Left against medical advice'    : 3,
        'Expired / Hospice'              : 4,
    }

    diag1_circulatory  = st.checkbox("Primary Diagnosis: Circulatory")
    diag1_other        = st.checkbox("Primary Diagnosis: Other")
    diag3_respiratory  = st.checkbox("Third Diagnosis: Respiratory")

st.divider()

# Predict button
if st.button("🔍 Predict Readmission Risk", use_container_width=True):

    # Engineer same features from Step 4
    total_services        = num_lab_procedures + num_medications
    has_emergency_history = 1 if number_emergency > 0 else 0
    high_utilizer         = 1 if (number_inpatient >= 2 or number_emergency >= 2) else 0

    # diag one-hot encoded columns
    diag_1_4 = 1 if diag1_circulatory else 0   # Circulatory encoded as 4
    diag_1_8 = 1 if diag1_other else 0          # Other encoded as 8
    diag_3_7 = 1 if diag3_respiratory else 0    # Respiratory encoded as 7

    # Build input row — exact column order matching X_final
    input_data = pd.DataFrame([{
        'ever_readmitted'         : ever_readmitted_val,
        'total_encounters'        : total_encounters,
        'max_inpatient_history'   : max_inpatient_history,
        'mean_medications_history': mean_medications_history,
        'total_services'          : total_services,
        'num_lab_procedures'      : num_lab_procedures,
        'num_medications'         : num_medications,
        'time_in_hospital'        : time_in_hospital,
        'age'                     : age,
        'discharge_disposition_id': discharge_map[discharge_disposition],
        'number_inpatient'        : number_inpatient,
        'number_diagnoses'        : number_diagnoses,
        'diag_3_7'                : diag_3_7,
        'metformin'               : metformin_map[metformin],
        'race_2'                  : race_2_val,
        'number_emergency'        : number_emergency,
        'diag_1_8'                : diag_1_8,
        'high_utilizer'           : high_utilizer,
        'diag_1_4'                : diag_1_4,
        'has_emergency_history'   : has_emergency_history,
    }])

    # Scale & predict
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ HIGH RISK — This patient is likely to be readmitted within 30 days")
    else:
        st.success("✅ LOW RISK — This patient is unlikely to be readmitted within 30 days")

    st.metric("Readmission Probability", f"{probability*100:.1f}%")
    st.progress(float(probability))

    st.divider()
    st.subheader("Risk Factors Summary")
    factors = {
        "Prior inpatient visits" : number_inpatient,
        "Prior emergency visits" : number_emergency,
        "Number of diagnoses"    : number_diagnoses,
        "Total medications"      : num_medications,
        "Previously readmitted"  : "Yes" if ever_readmitted_val else "No",
        "High utilizer"          : "Yes" if high_utilizer else "No",
    }
    for factor, value in factors.items():
        st.write(f"• **{factor}**: {value}")
        