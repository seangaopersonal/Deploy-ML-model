# Put the code for your API here.
import fastapi
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import pickle
from starter.ml.data import *
from starter.ml.model import *
import pandas as pd
import numpy as np

        
# Initialize API object
app = FastAPI()

# Load model

# rf_model_path = "model/rf_model.pkl"
# encoder_path = "model/encoder.pkl"
# lb_path = "model/labelbinarizer.pkl"

rf_model_path = "/Users/seangao/Desktop/Deploy-ML-model/starter/model/rf_model.pkl"
encoder_path = "/Users/seangao/Desktop/Deploy-ML-model/starter/model/encoder.pkl"
lb_path = "/Users/seangao/Desktop/Deploy-ML-model/starter/model/labelbinarizer.pkl"

with open(rf_model_path, 'rb') as f:
    rf_model = pickle.load(f)

with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)

with open(lb_path, 'rb') as f:
    lb = pickle.load(f)

# Declare the data object with its components and their type.
class InputData(BaseModel):
    # age: int
    # workclass: str 
    # fnlgt: int
    # education: str
    # education_num: int
    # marital_status: str
    # occupation: str
    # relationship: str
    # race: str
    # sex: str
    # capital_gain: int
    # capital_loss: int
    # hours_per_week: int
    # native_country: str
    age: int = Field(None, example=50)
    workclass: str = Field(None, example='Self-emp-not-inc')
    fnlgt: int = Field(None, example=83311)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example="Married-civ-spouse")
    occupation: str = Field(None, example="Exec-managerial")
    relationship: str = Field(None, example='Husband')
    race: str = Field(None, example='White')
    sex: str = Field(None, example='Male')
    capital_gain: int = Field(None, example=0)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=13)
    native_country: str = Field(None, example='United-States')
    # class Config:
    #     schema_extra = {
    #         "example": {
    #             "age": 50,
    #             "workclass": 'Self-emp-not-inc',
    #             "fnlgt": 83311,
    #             "education": 'Bachelors',
    #             "education_num": 13,
    #             "marital_status": "Married-civ-spouse",
    #             "occupation": "Exec-managerial",
    #             "relationship": "Husband",
    #             "race": "White",
    #             "sex": "Male",
    #             "capital_gain": 0,
    #             "capital_loss": 0,
    #             "hours_per_week": 13,
    #             "native_country": 'United-States'
    #         }
    #     }

@app.get('/')
async def welcome():
    return "Welcome to Sean Gao RF model API"


@app.post('/predict-income-class')
async def predict(data: InputData):

    # iterate through input data entry to create the respective datafrmae
    data_input_dict = {
        "age": data.age,
        "workclass" : data.workclass,
        "fnlwgt":  data.fnlgt,
        "education": data.education,
        "education_num": data.education_num,
        "marital-status": data.marital_status,
        "occupation":  data.occupation,
        "relationship": data.relationship,
        "race": data.race,
        "sex": data.sex,
        "capital_gain": data.capital_gain,
        "capital_loss": data.capital_loss,
        "hours-per-week": data.hours_per_week,
        "native-country": data.native_country}
    final_dict = {}
    for key,val in data_input_dict.items():
        final_dict[key] = [val]
    df_input = pd.DataFrame.from_dict(final_dict)

    cat_features = [
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native-country",
                ]

    # df_input = pd.DataFrame.from_dict(final_dict)
    X, _, _, _ = process_data(
        df_input,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    y_pred = inference(model=rf_model, X=X)
    return {"income_prediction": '<=50K' if y_pred == 0 else '>50K'}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8800, reload=True)