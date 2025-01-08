import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

#initialze FastAPI
app = FastAPI()

 #Body format for predictions
class PredictionFeatures(BaseModel):
    Ship_Mode: int
    Region: int
    Category: int
    Sub_Category: int
    Quantity: int

#Global variable to store the loaded model
model = None
scaler = None
poly = None

#Download the model
def download_model():
    global model
    global scaler
    global poly
    model = joblib.load('model.sav')
    scaler = joblib.load('std_scaler.sav')
    poly = joblib.load('poly_feat.sav')

download_model()

#API Root endpoint
@app.get("/")
async def index():
    return {"Welcome, Use the /predict feature to predict sales."}

# Prediction endpoint
@app.post("/predict")
async def predict(features: PredictionFeatures):

    # Convert input features to a NumPy array
    input_data = np.array([[
        features.Ship_Mode,
        features.Region,
        features.Category,
        features.Sub_Category,
        features.Quantity
    ]])

    # Preprocessing and predicting the loaded model
    input_data = scaler.transform(input_data)
    input_data = poly.transform(input_data)
    prediction = model.predict(input_data)[0]

    # Ensure the prediction is in a JSON-serializable format
    prediction = float(prediction)

    return { 'Price of Sales (USD)': prediction }

if __name__ == '__main__':
    uvicorn.run(app, host = "127.0.0.1", port = 8000)