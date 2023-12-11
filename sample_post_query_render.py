import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


features = {
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


app_url = "https://udacity-fhjr.onrender.com/predict-income-class"

r = requests.post(app_url, json=features)
assert r.status_code == 200

logging.info("Test Heroku app using sample input")
logging.info(f"Prediction output: {r.json()}")
