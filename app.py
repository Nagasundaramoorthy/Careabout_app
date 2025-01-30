from flask import Flask, render_template, request
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

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    risk_factors = []
    img_data = None

    if request.method == 'POST':
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        blood_sugar = float(request.form['blood_sugar'])
        cholesterol = float(request.form['cholesterol'])
        crp = float(request.form['crp'])
        renal_function = float(request.form['renal_function'])
        systolic = int(request.form['systolic'])
        diastolic = int(request.form['diastolic'])

        # Prepare input for prediction
        input_features = np.array([[age, bmi, blood_sugar, cholesterol, crp, renal_function, systolic, diastolic]])
        prediction = ensemble_model.predict(input_features)[0]
        risk_status = "At Risk" if prediction == 1 else "Not at Risk"

        # Identify risk factors
        if bmi > thresholds["BMI"]:
            risk_factors.append("BMI is above the healthy range. Consider a balanced diet and regular exercise.")
        if blood_sugar > thresholds["Blood Sugar"]:
            risk_factors.append("Blood sugar is high. Reduce sugar intake and monitor glucose levels.")
        if cholesterol > thresholds["Cholesterol"]:
            risk_factors.append("Cholesterol level is high. Eat fiber-rich foods and reduce saturated fats.")
        if crp > thresholds["CRP"]:
            risk_factors.append("CRP is elevated, indicating inflammation. Consider an anti-inflammatory diet.")
        if renal_function < thresholds["Renal Function"]:
            risk_factors.append("Renal function is low. Stay hydrated and monitor kidney health.")
        if systolic > thresholds["Systolic"] or diastolic > thresholds["Diastolic"]:
            risk_factors.append("Blood pressure is high. Reduce salt intake and practice stress management.")

        # Generate plot
        categories = list(thresholds.keys())
        values = [age, bmi, blood_sugar, cholesterol, crp, renal_function, systolic, diastolic]
        normal_values = [min(value, thresholds[cat]) for value, cat in zip(values, categories)]
        exceeding_values = [max(value - thresholds[cat], 0) for value, cat in zip(values, categories)]

        fig, ax = plt.subplots()
        ax.bar(categories, normal_values, color="skyblue", label="Normal Range")
        ax.bar(categories, exceeding_values, bottom=normal_values, color="red", label="Above Normal")
        ax.set_ylabel("Values")
        ax.set_title("Health Metrics")
        ax.legend()
        plt.xticks(rotation=90)
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_data = base64.b64encode(img.getvalue()).decode()
        plt.close()

    return render_template('index.html', prediction=prediction, risk_factors=risk_factors, img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)
