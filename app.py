from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('boston_housing_model.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI()

class HouseFeatures(BaseModel):
    features: list

@app.post('/predict')
def predict_price(data: HouseFeatures):
    features = np.array(data.features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    return {'predicted_price': prediction[0]}
