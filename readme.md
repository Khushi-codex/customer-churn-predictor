# Customer Churn Prediction API

Predicts whether a telecom customer will churn using Machine Learning — served as a live REST API.

🔴 **Live API:** https://customer-churn-predictor-cwzx.onrender.com/docs  
📂 **GitHub:** https://github.com/Khushi-codex/customer-churn-predictor

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 81% |
| F1 Score | 78% |
| AUC-ROC | 0.85 |
| Dataset size | 7,043 customers |

---

## What it does

Send customer details → get back churn probability + risk level (High / Medium / Low)

```json
POST /predict

{
  "tenure": 12,
  "MonthlyCharges": 65.5,
  "TotalCharges": 780.0,
  "Contract": 0,
  "InternetService": 1,
  "PaymentMethod": 2,
  "OnlineSecurity": 0,
  "TechSupport": 0
}
```

Response:
```json
{
  "churn_prediction": true,
  "churn_probability": 0.73,
  "risk_level": "High"
}
```

---

## Tech Stack

`Python` `Scikit-learn` `Pandas` `FastAPI` `Render`

---

## Run locally

```bash
git clone https://github.com/Khushi-codex/customer-churn-predictor.git
cd customer-churn-predictor
pip install -r requirements.txt
uvicorn main:app --reload
```

Open → http://127.0.0.1:8000/docs

---

## Key Finding

Customers on **month-to-month contracts churn 3x more** than those on 2-year contracts.

---
