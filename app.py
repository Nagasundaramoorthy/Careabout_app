from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)

# Load the model
with open('ensemble_model2.pkl', 'rb') as file:
    ensemble_model = pickle.load(file)

# Define thresholds
thresholds = {
    "Age": 120,
    "BMI": 25,
    "Blood Sugar": 140,
    "Cholesterol": 200,
    "CRP": 3,
    "Renal Function": 80,
    "Systolic": 120,
    "Diastolic": 80
}

@app.route('/', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract features
        age = data["age"]
        bmi = data["bmi"]
        blood_sugar = data["blood_sugar"]
        cholesterol = data["cholesterol"]
        crp = data["crp"]
        renal_function = data["renal_function"]
        systolic = data["systolic"]
        diastolic = data["diastolic"]

        # Prepare input for prediction
        input_features = np.array([[age, bmi, blood_sugar, cholesterol, crp, renal_function, systolic, diastolic]])
        prediction = ensemble_model.predict(input_features)[0]
        risk_status = "At Risk" if prediction == 1 else "Not at Risk"

        # Identify risk factors
        risk_factors = []
        if bmi > thresholds["BMI"]:
            risk_factors.append("BMI is above the healthy range.")
        if blood_sugar > thresholds["Blood Sugar"]:
            risk_factors.append("Blood sugar is high.")
        if cholesterol > thresholds["Cholesterol"]:
            risk_factors.append("Cholesterol level is high.")
        if crp > thresholds["CRP"]:
            risk_factors.append("CRP is elevated, indicating inflammation.")
        if renal_function < thresholds["Renal Function"]:
            risk_factors.append("Renal function is low.")
        if systolic > thresholds["Systolic"] or diastolic > thresholds["Diastolic"]:
            risk_factors.append("Blood pressure is high.")

        # Return JSON response
        return jsonify({
            "prediction": risk_status,
            "risk_factors": risk_factors
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
