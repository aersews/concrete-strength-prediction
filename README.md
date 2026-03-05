

---

# Concrete Compressive Strength Predictor

A reproducible **machine learning pipeline** for predicting concrete compressive strength (MPa) using 14 regression models, structured evaluation, and hyperparameter tuning.
The best model is deployed via a lightweight Flask web interface.

**Best Result:**
**CatBoost (Tuned)** → R² = **0.9395**, MAE = **2.55 MPa**

---

## 🔍 Features

* 14 baseline regression models
* 5-fold cross-validation
* RandomizedSearchCV hyperparameter tuning
* Strict held-out test set (no leakage)
* Leaderboard export (CSV)
* Prediction vs. actual & feature importance plots
* Flask + AJAX prediction UI
* Fully reproducible (fixed random seed)

---

## 📁 Structure

```
.
├── main.py                 # Training & evaluation pipeline
├── app.py                  # Flask web app
├── Concrete_Data.xls       # Dataset
├── templates/index.html
├── outputs/                # Metrics, plots, best_model.pkl
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### 2. Train models

```bash
python main.py
```

Outputs saved to:

```
outputs/
```

### 3. Launch web app

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## 📊 Models Compared

Linear, Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost, SVR, KNN, MLP

Tuned models:

* CatBoost
* RandomForest

---

## 📈 Evaluation Metrics

* MAE
* RMSE
* R²
* Adjusted R²

---

## 🏆 Final Leaderboard (Top Models)

| Rank | Model          | Test R²    |
| ---- | -------------- | ---------- |
| 1    | CatBoost_Tuned | **0.9395** |
| 2    | CatBoost       | 0.9343     |
| 3    | LightGBM       | 0.9297     |
| 4    | XGBoost        | 0.9231     |

Full results available in:

```
outputs/final_leaderboard.csv
```

---

## 🔒 Reproducibility

* Fixed random seed (42)
* Strict test-set isolation
* No early stopping leakage in CV
* Cleaned parameter distributions
* Consistent evaluation tables

---

## 🛠 Future Improvements

* SHAP explainability
* Docker container
* FastAPI REST API
* Learning curves
* CLI batch prediction

---

A clean, research-oriented regression benchmark project suitable for experimentation, extension, or deployment.

---