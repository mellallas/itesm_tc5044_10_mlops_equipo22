import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import uvicorn
import yaml
from sklearn.datasets import load_wine

# Load the saved model
with open("models/rfc_model.pkl", "rb") as f:
    model = pickle.load(f)

with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)

# Load target names for response
data = pd.read_csv(config['data']['input_data'])
target_names = data['Load_Type'].unique()

# Define the input data format for prediction
class SteelIndustryData(BaseModel):
    features: List[float]

# Initialize FastAPI app
app = FastAPI()

# Define prediction endpoint
@app.post("/predict")
def predict(steelind_data: SteelIndustryData):
    # Validate input length
    if len(steelind_data.features) != model.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Input must contain {model.n_features_in_} features."
        )

    # Make prediction
    prediction = model.predict([steelind_data.features])[0]
    prediction_name = target_names[prediction]
    
    return {"prediction": int(prediction), "prediction_name": prediction_name}

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Steel Industry classification model API"}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
