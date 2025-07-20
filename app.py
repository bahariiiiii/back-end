from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import xgboost as xgb
import os
#http://nl.smait.ir

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Load the trained model - ensure this file exists in the same folder as app.py
model = xgb.XGBClassifier()
model.load_model("xgboost_model2.json")

def to_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def preprocess_input(data):
    # Convert categorical field 'otherInfections' to binary 0/1
    has_other = data.get("otherInfections", "").lower()
    has_other_infections_yes = 1 if has_other == "yes" else 0

    features = [
        to_float(data.get("age")),
        to_float(data.get("bmi")),
        to_float(data.get("apache_score")),
        to_float(data.get("heart_rate")),
        to_float(data.get("respiratory_rate")),
        to_float(data.get("body_temperature")),
        to_float(data.get("systolic_pressure")),
        to_float(data.get("diastolic_pressure")),
        to_float(data.get("gcs_score")),
        to_float(data.get("urine_output")),
        to_float(data.get("na")),
        to_float(data.get("k")),
        to_float(data.get("hco3")),
        to_float(data.get("creatinine")),
        to_float(data.get("hemoglobin")),
        to_float(data.get("hematocrit")),
        to_float(data.get("albumin")),
        to_float(data.get("ph")),
        to_float(data.get("wbc")),
        to_float(data.get("platelet")),
        to_float(data.get("paO2_fio2")),
        to_float(data.get("bilirubin")),
        to_float(data.get("blood_sugar")),
        to_float(data.get("alk")),
        to_float(data.get("alt")),
        to_float(data.get("ast")),
        to_float(data.get("mch")),
        to_float(data.get("mcv")),
        to_float(data.get("mchc")),
        to_float(data.get("bun")),
        to_float(data.get("rdw")),
        to_float(data.get("pt")),
        to_float(data.get("numberOfCatheters")),
        to_float(data.get("daysOfCatheterization")),
        has_other_infections_yes
    ]

    return np.array(features).reshape(1, -1)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    try:
        X = preprocess_input(data)
        proba = float(model.predict_proba(X)[0, 1])  # Convert numpy float32 to native float
        risk_percent = round(proba * 100, 2)
        return jsonify({"risk": risk_percent})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
