import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, Form, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib

# Initialize FastAPI
app = FastAPI()

# Set up Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="templates")

# Body format for predictions
class PredictionFeatures(BaseModel):
    Ship_Mode: int
    Region: int
    Sub_Category: int
    Quantity: int

# Global variable to store the loaded model
model = None
scaler = None
poly = None

# Download the model
def download_model():
    global model
    global scaler
    global poly
    model = joblib.load('model.sav')
    scaler = joblib.load('std_scaler.sav')

download_model()

# API Root endpoint
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("front4.html", {"request": request})

# Prediction endpoint
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  Ship_Mode: int = Form(...),
                  Region: int = Form(...),
                  Sub_Category: int = Form(...),
                  Quantity: int = Form(...)):

    # Convert input features to a NumPy array
    input_data = np.array([[Ship_Mode, Region, Sub_Category, Quantity]])

    # Preprocessing and predicting with the loaded model
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)[0]

    result = f" Sales: {prediction:.2f} $"
    prediction = result

    return templates.TemplateResponse("front4.html", {"request": request, "prediction": prediction})

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
