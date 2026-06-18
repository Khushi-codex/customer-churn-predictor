from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_valid():
    payload = {"tenure": 2, "MonthlyCharges": 95.0, "TotalCharges": 190.0,
               "Contract": 0, "InternetService": 1, "PaymentMethod": 2,
               "OnlineSecurity": 0, "TechSupport": 0}
    r = client.post("/predict", json=payload)
    assert r.status_code in [200, 503]
    if r.status_code == 200:
        d = r.json()
        assert "churn_prediction" in d
        assert "churn_probability" in d
        assert d["risk_level"] in ["High", "Medium", "Low"]
        assert 0.0 <= d["churn_probability"] <= 1.0

def test_invalid_input():
    r = client.post("/predict", json={"tenure": -5})
    assert r.status_code == 422
