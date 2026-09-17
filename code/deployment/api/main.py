import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Wine Quality API")

bundle = joblib.load("models/model.joblib")

model = bundle["model"]
scaler = bundle["scaler"]


class Wine(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


@app.get("/")
def root():
    return {"message": "Wine Quality API is running"}


@app.post("/predict")
def predict(wine: Wine):
    data = pd.DataFrame([{
        "fixed acidity": wine.fixed_acidity,
        "volatile acidity": wine.volatile_acidity,
        "citric acid": wine.citric_acid,
        "residual sugar": wine.residual_sugar,
        "chlorides": wine.chlorides,
        "free sulfur dioxide": wine.free_sulfur_dioxide,
        "total sulfur dioxide": wine.total_sulfur_dioxide,
        "density": wine.density,
        "pH": wine.pH,
        "sulphates": wine.sulphates,
        "alcohol": wine.alcohol,
    }])

    features = scaler.transform(data)

    prediction = int(model.predict(features)[0])

    return {
        "prediction": prediction,
        "label": "Good wine" if prediction == 1 else "Not good wine",
    }