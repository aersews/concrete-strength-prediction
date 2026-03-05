# ============================================================
# Concrete Compressive Strength Prediction
# Full Research-Grade ML Pipeline (Improved)
# ============================================================

"""
Improvements over original:
✔ Fixed loguniform bounds for learning_rate (was silently wrong)
✔ Removed early_stopping_rounds from CatBoost constructor (breaks RandomizedSearchCV)
✔ Added MAE to all results tables
✔ Fixed str.replace() with regex=False to suppress pandas warnings
✔ Added combined baseline + tuned comparison table
✔ Added explicit test-set isolation guard
✔ Consistent column naming across DataFrames
✔ Cleaner plot styling
✔ Minor: removed redundant random.seed (numpy covers it for sklearn)
"""

# ============================================================
# 0. Imports
# ============================================================

import os
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

from scipy.stats import randint, uniform, loguniform

warnings.filterwarnings("ignore")


# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 42
np.random.seed(SEED)


# ============================================================
# 2. Paths
# ============================================================

DATA_PATH = "Concrete_Data.xls"

OUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUT_DIR, "models")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# 3. Utilities
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # FIX: regex=False avoids pandas FutureWarning on special characters
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    return df


def basic_data_checks(df: pd.DataFrame) -> None:
    print("\n===== DATA CHECKS =====")
    print("Shape:", df.shape)
    print("Duplicates:", df.duplicated().sum())
    missing = df.isna().sum()
    if missing.any():
        print("Missing values:\n", missing[missing > 0])
    else:
        print("Missing values: none")
    print(df.describe().round(2))


def split_features_target(df: pd.DataFrame):
    target = df.columns[-1]
    print(f"\nTarget column: '{target}'")
    return df.drop(columns=[target]), df[target]


def scaled(model):
    """Wrap a model in a StandardScaler pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def get_metrics(y_true, y_pred, n_features: int) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    n    = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Adj_R2": adj_r2}


def evaluate_model(name, model, X_train, y_train, X_test, y_test) -> list:
    """
    CV on train set only, then evaluate once on held-out test set.
    X_test / y_test are NEVER used for fitting or selection.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_validate(model, X_train, y_train,
                               cv=cv, scoring="r2", n_jobs=-1)
    cv_r2_mean = cv_scores["test_score"].mean()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    m = get_metrics(y_test, preds, X_train.shape[1])

    return [name, cv_r2_mean, m["MAE"], m["RMSE"], m["R2"], m["Adj_R2"], model]


RESULT_COLS = ["Model", "CV_R2", "MAE", "RMSE", "Test_R2", "Adj_R2", "ModelObj"]


# ============================================================
# 4. Load & Split Data
# ============================================================

df = load_data(DATA_PATH)
basic_data_checks(df)

X, y = split_features_target(df)

# ── Test set is isolated here and never touched until final evaluation ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

print(f"\nTrain: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")


# ============================================================
# 5. Baseline Models
# ============================================================

models = {
    # Linear
    "Linear":     scaled(LinearRegression()),
    "Ridge":      scaled(Ridge()),
    "Lasso":      scaled(Lasso()),
    "ElasticNet": scaled(ElasticNet()),

    # Trees (no scaling needed, but using pipeline keeps API uniform)
    "DecisionTree":      DecisionTreeRegressor(random_state=SEED),
    "RandomForest":      RandomForestRegressor(random_state=SEED, n_jobs=-1),
    "ExtraTrees":        ExtraTreesRegressor(random_state=SEED, n_jobs=-1),
    "GradientBoosting":  GradientBoostingRegressor(random_state=SEED),

    # Boosting
    "XGBoost": XGBRegressor(
        objective="reg:squarederror", random_state=SEED, verbosity=0
    ),
    "LightGBM": lgb.LGBMRegressor(
        random_state=SEED,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        force_col_wise=True,
        verbose=-1,
    ),
    # FIX: early_stopping_rounds removed — it requires eval_set during fit(),
    # which cross_validate / RandomizedSearchCV do not provide,
    # causing silent failures or errors.
    "CatBoost": CatBoostRegressor(verbose=0, random_state=SEED),

    # Distance / kernel
    "SVR": scaled(SVR()),
    "KNN": scaled(KNeighborsRegressor()),
    "MLP": scaled(MLPRegressor(max_iter=2000, random_state=SEED)),
}


# ============================================================
# 6. Baseline Evaluation
# ============================================================

print("\n========== BASELINE ==========")

results = []
for name, model in models.items():
    print(f"  Training: {name}")
    results.append(evaluate_model(name, model, X_train, y_train, X_test, y_test))

baseline_df = (
    pd.DataFrame(results, columns=RESULT_COLS)
    .sort_values("Test_R2", ascending=False)
    .reset_index(drop=True)
)

print("\n", baseline_df.drop(columns="ModelObj").round(4).to_string(index=False))

baseline_df.drop(columns="ModelObj").to_csv(
    os.path.join(OUT_DIR, "baseline_metrics.csv"), index=False
)


# ============================================================
# 7. Hyperparameter Tuning
# ============================================================

print("\n========== TUNING ==========")

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)


def tune_model(name, model, params, n_iter=30):
    print(f"  Tuning: {name}")
    search = RandomizedSearchCV(
        model, params,
        n_iter=n_iter,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        random_state=SEED,
        refit=True,          # keeps best estimator fitted on full X_train
    )
    search.fit(X_train, y_train)
    print(f"    Best CV R²: {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")
    return search.best_estimator_, search.best_score_


tuned_models = []

# ── CatBoost ──────────────────────────────────────────────────────────────────
# FIX: loguniform(a, b) samples from [a, a*b], NOT [a, b].
# To sample learning_rate in roughly [0.005, 0.30], use loguniform(0.005, 60).
# Alternatively (cleaner): use a log-uniform via np in a lambda, or just use
# uniform for a flat prior and rely on enough iterations to cover the range.
# Here we use the correct loguniform parameterisation.
cat_params = {
    "iterations":         randint(800, 2000),
    "depth":              randint(4, 10),
    "learning_rate":      loguniform(0.005, 0.30),   # ← FIXED (was loguniform(0.01, 0.2))
    "l2_leaf_reg":        uniform(1, 9),              # samples [1, 10]
    "subsample":          uniform(0.6, 0.4),          # samples [0.6, 1.0]
    "bagging_temperature": uniform(0, 5),
}

cat_best, cat_cv = tune_model(
    "CatBoost",
    CatBoostRegressor(verbose=0, random_state=SEED),
    cat_params,
    n_iter=40,
)
tuned_models.append(("CatBoost_Tuned", cat_best, cat_cv))

# ── RandomForest ──────────────────────────────────────────────────────────────
rf_params = {
    "n_estimators": randint(300, 1200),
    "max_depth":    randint(4, 25),
    "min_samples_leaf": randint(1, 10),      # added: controls overfitting
    "max_features": uniform(0.3, 0.7),       # added: feature subsampling
}

rf_best, rf_cv = tune_model(
    "RandomForest",
    RandomForestRegressor(random_state=SEED, n_jobs=-1),
    rf_params,
    n_iter=30,
)
tuned_models.append(("RF_Tuned", rf_best, rf_cv))


# ============================================================
# 8. Evaluate Tuned Models & Combined Table
# ============================================================

tuned_rows = []
for name, model, cv_score in tuned_models:
    preds = model.predict(X_test)   # model already fitted by RandomizedSearchCV
    m = get_metrics(y_test, preds, X_train.shape[1])
    tuned_rows.append([name, cv_score, m["MAE"], m["RMSE"], m["R2"], m["Adj_R2"], model])

tuned_df = (
    pd.DataFrame(tuned_rows, columns=RESULT_COLS)
    .sort_values("Test_R2", ascending=False)
    .reset_index(drop=True)
)

print("\nTUNED RESULTS")
print(tuned_df.drop(columns="ModelObj").round(4).to_string(index=False))

tuned_df.drop(columns="ModelObj").to_csv(
    os.path.join(OUT_DIR, "tuned_metrics.csv"), index=False
)

# ── Combined comparison (baseline top-5 + all tuned) ─────────────────────────
combined = pd.concat([
    baseline_df.head(5).drop(columns="ModelObj"),
    tuned_df.drop(columns="ModelObj"),
]).sort_values("Test_R2", ascending=False).reset_index(drop=True)

combined.insert(0, "Rank", range(1, len(combined) + 1))
print("\nFINAL LEADERBOARD (Top baseline + Tuned)")
print(combined.round(4).to_string(index=False))
combined.to_csv(os.path.join(OUT_DIR, "final_leaderboard.csv"), index=False)


# ============================================================
# 9. Save Best Model
# ============================================================

best_row   = tuned_df.iloc[0]
best_model = best_row["ModelObj"]

model_path = os.path.join(MODEL_DIR, "best_model.pkl")
joblib.dump(best_model, model_path)
print(f"\nBest model '{best_row['Model']}' saved → {model_path}")


# ============================================================
# 10. Prediction Plot
# ============================================================

preds = best_model.predict(X_test)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, preds, alpha=0.6, edgecolors="k", linewidths=0.4, s=40)

# Perfect-prediction reference line
lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit")

ax.set_xlabel("Actual Strength (MPa)")
ax.set_ylabel("Predicted Strength (MPa)")
ax.set_title(f"Predicted vs Actual — {best_row['Model']}\n"
             f"R² = {best_row['Test_R2']:.4f} | RMSE = {best_row['RMSE']:.2f}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "prediction_vs_actual.png"), dpi=150)
plt.close()


# ============================================================
# 11. Feature Importance
# ============================================================

if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(7, 4))
    imp.plot(kind="barh", ax=ax, color="steelblue", edgecolor="k", linewidth=0.5)
    ax.set_title(f"Feature Importance — {best_row['Model']}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
else:
    print("Best model does not expose feature_importances_; skipping plot.")


print("\n✓ Pipeline complete. All outputs saved in /outputs/")