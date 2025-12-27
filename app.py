from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open(r"C:\Users\KARTHIK\OneDrive\Desktop\credit card fraud detection\fraud_model.pkl", "rb"))

scaler = pickle.load(open(r"C:\Users\KARTHIK\OneDrive\Desktop\credit card fraud detection\scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        v1 = float(request.form["v1"])
        v2 = float(request.form["v2"])
        v3 = float(request.form["v3"])
        amount = float(request.form["amount"])

        # Scale amount
        amount_scaled = scaler.transform([[amount]])[0][0]

        # Predict
        prob = model.predict_proba([[v1, v2, v3, amount_scaled]])[0][1]

        if prob > 0.1:
            prediction = f"⚠️ Fraud Transaction (Risk Score: {prob:.2f})"
        else:
            prediction = f"✅ Legitimate Transaction (Risk Score: {prob:.2f})"


    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
