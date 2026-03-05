# ============================================================
# Flask Backend — Modern AJAX Concrete Strength Predictor
# ============================================================

import os
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


# ------------------------------------------------------------
# Load model safely
# ------------------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "outputs",
    "models",
    "best_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ------------------------------------------------------------
# API Prediction (AJAX)
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        values = [
            float(data["cement"]),
            float(data["slag"]),
            float(data["flyash"]),
            float(data["water"]),
            float(data["sp"]),
            float(data["coarse"]),
            float(data["fine"]),
            float(data["age"]),
        ]

        sample = np.array([values])
        prediction = model.predict(sample)[0]

        return jsonify({
            "success": True,
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)