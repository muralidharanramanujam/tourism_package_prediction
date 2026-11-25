import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Tourism Package Predictor", layout="wide")

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write(
    """
    This application predicts whether a customer is likely to purchase the **Wellness Tourism Package**
    (`ProdTaken`) based on customer profile details and sales interaction attributes.
    """
)

st.subheader("Customer & Trip Details")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])

with col2:
    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=0.0, max_value=500000.0, value=25000.0, step=500.0
    )
    num_persons = st.number_input(
        "Number of Persons Visiting", min_value=1, max_value=20, value=2, step=1
    )
    num_children = st.number_input(
        "Number of Children Visiting (< 5 yrs)", min_value=0, max_value=10, value=0, step=1
    )
    num_trips = st.number_input(
        "Average Annual Trips", min_value=0, max_value=50, value=2, step=1
    )

with col3:
    preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    passport = st.selectbox("Has Valid Passport", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    own_car = st.selectbox("Owns a Car", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

st.subheader("Sales & Interaction Details")
icol1, icol2, icol3, icol4 = st.columns(4)

with icol1:
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
with icol2:
    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
    )
with icol3:
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=3, step=1)
with icol4:
    duration_of_pitch = st.number_input("Duration of Pitch (min)", min_value=1, max_value=180, value=15, step=1)

# Exact 18 predictor columns expected by the trained pipeline (CustomerID and ProdTaken excluded)
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_satisfaction,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_of_pitch,
}])

if st.button("Predict Package Purchase (ProdTaken)"):
    prediction = int(model.predict(input_data)[0])
    proba = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    result_label = (
        "Likely to Purchase Wellness Tourism Package (ProdTaken = 1)"
        if prediction == 1
        else "Unlikely to Purchase Wellness Tourism Package (ProdTaken = 0)"
    )

    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result_label}**")
    if proba is not None:
        st.caption(f"Estimated purchase probability: {proba:.2%}")
