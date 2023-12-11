"""
This module includes unit tests for the ML model

"""
from sklearn.model_selection import train_test_split
import logging 
from starter.ml.model import train_model, compute_model_metrics, inference
from starter.ml.data import process_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest 

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


temp = {
    'age':[4,50,500],
    'workclass':["A","B","A"], 
    'salary':[1,0,1]
}

df_input = pd.DataFrame.from_dict(temp)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df_input, test_size=0.50)

cat_features = [
    "workclass"
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",
     training=False, encoder=encoder, lb=lb)


@pytest.fixture(scope="module")
def model():
    # code to load in the data.
    rf_model = train_model(X_train, y_train)
    return rf_model



def test_train_model( ):
    
    """ Check training model to return rf classifier """
    rf_model = train_model(X_train, y_train)

    assert isinstance(rf_model, RandomForestClassifier)


def test_compute_model_metrics(model):

    """ Check model type """

    precision, recall, fbeta  = compute_model_metrics(y_train, inference(model, X_train))
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference(model):

    """ Test the data split """

    y_pred = inference(model, X_train)
    assert y_pred.shape[0] == y_train.shape[0]

def test_inference_2(model):

    """ Test the data split """

    y_pred = inference(model, X_train)
    assert y_pred in [0,1]