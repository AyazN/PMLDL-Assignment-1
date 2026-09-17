import requests
import streamlit as st


API_URL = "http://api:8000/predict"

st.title("Wine Quality Predictor")

st.write("Enter the wine characteristics:")

fixed_acidity = st.number_input("Fixed acidity", value=7.4)
volatile_acidity = st.number_input("Volatile acidity", value=0.7)
citric_acid = st.number_input("Citric acid", value=0.0)
residual_sugar = st.number_input("Residual sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free sulfur dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total sulfur dioxide", value=34.0)
density = st.number_input("Density", value=0.9978)
ph = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)


if st.button("Predict"):
    data = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": ph,
        "sulphates": sulphates,
        "alcohol": alcohol,
    }

    try:
        response = requests.post(API_URL, json=data)
        response.raise_for_status()

        result = response.json()

        st.subheader("Prediction")
        st.write(result["label"])

    except requests.RequestException as e:
        st.error(f"API error: {e}")