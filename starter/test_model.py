"""
This module includes unit tests for the ML model

"""
from sklearn.model_selection import train_test_split
import logging 
from .starter.ml.model import train_model, compute_model_metrics, inference
from .starter.ml.data import process_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest 

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

df_input = pd.read_csv("/starter/data/census_cleaned.csv", header = 0)


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df_input, test_size=0.20)

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

    precision, _, _ = compute_model_metrics(y_train, inference(model, X_train))
    assert isinstance(precision, float)


def test_inference(model, X_train):

    """ Test the data split """

    y_pred = inference(model, X_train)
    assert y_pred.shape[0] == y_train.shape[0]