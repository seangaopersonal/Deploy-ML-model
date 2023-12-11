# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import logging 
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Add the necessary imports for the starter code.


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Add code to load in the data.
logging.info("loading the cleaned census data")
df_input = pd.read_csv("/Users/seangao/Desktop/Deploy-ML-model/starter/data/census_cleaned.csv", header = 0)


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

logging.info("Starting model training")
rf_model = train_model(X_train, y_train)

logging.info("Saving artifacts")
pickle.dump(rf_model, open("/Users/seangao/Desktop/Deploy-ML-model/starter/model/rf_model.pkl", 'wb'))
pickle.dump(encoder, open("/Users/seangao/Desktop/Deploy-ML-model/starter/model/encoder.pkl", 'wb'))
pickle.dump(lb, open("/Users/seangao/Desktop/Deploy-ML-model/starter/model/labelbinarizer.pkl", 'wb'))

logging.info("Calculating test set model metrics")
y_pred = inference(rf_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
logging.info(f"Model metrics output - Precision score: {precision}. Recall: {recall}. Fbeta: {fbeta}")