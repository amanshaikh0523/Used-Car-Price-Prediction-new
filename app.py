import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# -----------------------------
# Load Files
# -----------------------------
model = joblib.load("used_car_price_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Load dataset for dropdown values
df = pd.read_csv("usedCars (2).csv")

# -----------------------------
# Title
# -----------------------------
st.title("🚗 Used Car Price Prediction")

st.write(
    "Enter the car details below to predict its estimated resale price."
)

st.write("---")

# -----------------------------
# Input Fields
# -----------------------------

company = st.selectbox(
    "Select Company",
    sorted(df["Company"].unique())
)

company_df = df[df["Company"] == company]

model_name = st.selectbox(
    "Select Model",
    sorted(company_df["Model"].unique())
)

model_df = company_df[company_df["Model"] == model_name]

variant = st.selectbox(
    "Select Variant",
    sorted(model_df["Variant"].unique())
)

fuel_type = st.selectbox(
    "Fuel Type",
    sorted(df["FuelType"].unique())
)

body_style = st.selectbox(
    "Body Style",
    sorted(df["BodyStyle"].unique())
)

transmission = st.selectbox(
    "Transmission",
    sorted(df["TransmissionType"].unique())
)

owner = st.selectbox(
    "Owner",
    sorted(df["Owner"].unique())
)

model_year = st.number_input(
    "Model Year",
    min_value=2000,
    max_value=datetime.now().year,
    value=2020
)

kilometer = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=30000
)

warranty = st.selectbox(
    "Warranty Available",
    ["Yes", "No"]
)

quality_score = st.slider(
    "Quality Score",
    min_value=0.0,
    max_value=10.0,
    value=8.0,
    step=0.1
)

st.write("---")

predict = st.button("Predict Price")

# ----------------------------------
# Prediction
# ----------------------------------

if predict:

    # Create a dataframe with all model features
    input_df = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    # -----------------------------
    # Numerical Features
    # -----------------------------

    current_year = datetime.now().year

    car_age = current_year - model_year

    if car_age == 0:
        mileage_per_year = kilometer
    else:
        mileage_per_year = kilometer / car_age

    is_automatic = 1 if transmission.lower() == "automatic" else 0

    luxury_brands = [
        "BMW",
        "MERCEDES-BENZ",
        "AUDI",
        "VOLVO",
        "LEXUS",
        "JAGUAR",
        "LAND ROVER",
        "PORSCHE"
    ]

    luxury_brand = 1 if company.upper() in luxury_brands else 0

    # Store numerical values

    if "Kilometer" in input_df.columns:
        input_df["Kilometer"] = kilometer

    if "Warranty" in input_df.columns:
        input_df["Warranty"] = 1 if warranty == "Yes" else 0

    if "QualityScore" in input_df.columns:
        input_df["QualityScore"] = quality_score

    if "CarAge" in input_df.columns:
        input_df["CarAge"] = car_age

    if "MileagePerYear" in input_df.columns:
        input_df["MileagePerYear"] = mileage_per_year

    if "IsAutomatic" in input_df.columns:
        input_df["IsAutomatic"] = is_automatic

    if "LuxuryBrand" in input_df.columns:
        input_df["LuxuryBrand"] = luxury_brand

    # -----------------------------
    # One Hot Encoding
    # -----------------------------

    company_col = "Company_" + company

    if company_col in input_df.columns:
        input_df[company_col] = 1

    model_col = "Model_" + model_name

    if model_col in input_df.columns:
        input_df[model_col] = 1

    variant_col = "Variant_" + variant

    if variant_col in input_df.columns:
        input_df[variant_col] = 1

    fuel_col = "FuelType_" + fuel_type

    if fuel_col in input_df.columns:
        input_df[fuel_col] = 1

    body_col = "BodyStyle_" + body_style

    if body_col in input_df.columns:
        input_df[body_col] = 1

    transmission_col = "TransmissionType_" + transmission

    if transmission_col in input_df.columns:
        input_df[transmission_col] = 1

    owner_col = "Owner_" + owner

    if owner_col in input_df.columns:
        input_df[owner_col] = 1


    # -----------------------------
    # Scale Data
    # -----------------------------

    input_scaled = scaler.transform(input_df)

        # -----------------------------
    # Predict Price
    # -----------------------------

    prediction = model.predict(input_scaled)

    predicted_price = prediction[0]

    # -----------------------------
    # Display Prediction
    # -----------------------------

    st.success("Prediction Completed Successfully!")

    st.subheader("Estimated Used Car Price")

    st.metric(
        label="Predicted Price",
        value=f"₹ {predicted_price:,.0f}"
    )

    st.write("---")

    # -----------------------------
    # Car Details
    # -----------------------------

    st.subheader("Car Details")

    details = pd.DataFrame({

        "Feature":[
            "Company",
            "Model",
            "Variant",
            "Fuel Type",
            "Body Style",
            "Transmission",
            "Model Year",
            "Kilometers Driven",
            "Owner",
            "Warranty",
            "Quality Score"
        ],

        "Value":[
            company,
            model_name,
            variant,
            fuel_type,
            body_style,
            transmission,
            model_year,
            kilometer,
            owner,
            warranty,
            quality_score
        ]

    })

    st.table(details)

    st.write("---")

    st.info(
        "This prediction is generated using a Ridge Regression model trained on historical used car data."
    )

    # ----------------------------------
# Footer
# ----------------------------------

st.write("---")

st.markdown(
    """
    ### About Project

    This application predicts the resale price of a used car using a Machine Learning model.

    **Algorithm Used:** Ridge Regression

    **Developed By:** Aman Shaikh
    """
)