from fastapi.testclient import TestClient
from main import app
client = TestClient(app)


def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() ==  "Welcome to Sean Gao RF model API"


def test_post_predict():
    r = client.post("/predict-income-class", json={
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
    })
    assert r.status_code == 200
    assert r.json() == {"income_prediction": "<=50K"}

def test_post_predict_2():
    r = client.post("/predict-income-class", json={
        "age": 37,
        "workclass": "Private",
        "fnlgt": 280464,
        "education": "Some-college",
        "education_num": 15,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 80,
        "native_country": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"income_prediction": ">50K"}