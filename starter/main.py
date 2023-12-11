# Put the code for your API here.
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pickle
import os
from starter.ml.data import *
from starter.ml.model import *
import pandas as pd
import numpy as np

        
# Initialize API object
app = FastAPI()

# Load model

rf_model_path = "starter/model/rf_model.pkl"
encoder_path = "starter/model/encoder.pkl"
lb_path = "starter/model/labelbinarizer.pkl"

with open(rf_model_path, 'rb') as f:
    rf_model = pickle.load(f)

with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)

with open(lb_path, 'rb') as f:
    lb = pickle.load(f)

# Declare the data object with its components and their type.
class InputData(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    class Config:
        schema_extra = {
            "example": {
                "age": 50,
                "workclass": 'Self-emp-not-inc',
                "fnlgt": 83311,
                "education": 'Bachelors',
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 13,
                "native_country": 'United-States'
            }
        }

@app.get('/')
async def welcome():
    return "Welcome to RF model API"


@app.post('/predict-income-class')
async def predict(data: InputData):

    # iterate through input data entry to create the respective datafrmae
    data_input = np.array([[
        input.age,
        input.workclass,
        input.fnlgt,
        input.education,
        input.education_num,
        input.marital_status,
        input.occupation,
        input.relationship,
        input.race,
        input.sex,
        input.capital_gain,
        input.capital_loss,
        input.hours_per_week,
        input.native_country]])

    feature_col = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours-per-week",
        "native-country"]

    df_input = pd.DataFrame(data_input, columns = feature_col)
    X, _, _, _ = process_data(
        df_input,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    output = inference(model=model, X=X)[0]
    str_out = '<=50K' if output == 0 else '>50K'
    return {"pred": str_out}


if __name__ == '__main__':
    config = uvicorn.config("main:app", host="0.0.0.0",
                            reload=True, port=8080, log_level="info")
    server = uvicorn.Server(config)
    server.run()