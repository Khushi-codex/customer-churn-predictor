# Customer Churn Prediction API

An end-to-end machine learning system that predicts customer churn using the Telco Customer Churn dataset (7,043 records, 21 features). Trained a Random Forest classifier and served it as a live REST API.

**Live Demo:** `https://churn-predictor/docs`  
**Dataset:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | ~81% |
| F1 Score | ~78% |
| AUC-ROC | ~85% |
| Test set size | 1,409 customers |

*(Update these with your actual numbers after training)*

---

## Project Structure

```
churn-predictor/
├── data/
│   └── churn.csv              # Telco dataset (download from Kaggle)
├── notebooks/
│   └── analysis.ipynb         # EDA + model training + evaluation
├── model/
│   └── churn_model.pkl        # Saved trained model
├── tests/
│   └── test_api.py            # API endpoint tests
├── main.py                    # FastAPI application
├── requirements.txt
└── README.md
```

---

## Setup & Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/churn-predictor.git
cd churn-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Go to [Kaggle Telco Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), download `WA_Fn-UseC_-Telco-Customer-Churn.csv`, and save it as `data/churn.csv`.

**4. Train the model**
```bash
jupyter notebook notebooks/analysis.ipynb
# Run all cells — this saves model/churn_model.pkl
```

**5. Start the API**
```bash
uvicorn main:app --reload
```

**6. Open the interactive docs**
```
http://127.0.0.1:8000/docs
```

---

## API Usage

### `GET /`
Health check.

```json
{ "message": "Churn Prediction API is live" }
```

### `POST /predict`
Send customer data, get churn probability.

**Request body:**
```json
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

**Response:**
```json
{
  "churn_prediction": true,
  "churn_probability": 0.73,
  "risk_level": "High"
}
```

**Field reference:**

| Field | Values |
|-------|--------|
| `Contract` | 0 = Month-to-month, 1 = One year, 2 = Two year |
| `InternetService` | 0 = DSL, 1 = Fiber optic, 2 = No internet |
| `PaymentMethod` | 0 = Bank transfer, 1 = Credit card, 2 = Electronic check, 3 = Mailed check |
| `OnlineSecurity` | 0 = No, 1 = Yes |
| `TechSupport` | 0 = No, 1 = Yes |

---

## Key Findings from EDA

- Customers on **month-to-month contracts** churn at 3x the rate of annual contracts
- **Fiber optic** internet users have higher churn than DSL users despite faster speeds
- Customers with **no online security or tech support** are significantly more likely to churn
- Average **tenure of churned customers** is 18 months vs 37 months for retained customers

*(Fill this in with your actual findings from the notebook)*

---

## Tech Stack

- **ML:** Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **API:** FastAPI, Pydantic, Uvicorn
- **Deployment:** Render (free tier)
- **Model:** Random Forest Classifier (100 estimators)

---

## What I Learned

- Handling real data quality issues (TotalCharges stored as string, not float)
- Dealing with class imbalance (74% no-churn vs 26% churn)
- The difference between accuracy and AUC-ROC as evaluation metrics
- Serving a scikit-learn model via FastAPI with Pydantic validation
- Deploying a Python API to production on Render