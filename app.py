from flask import Flask, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("cars_model.pkl", "rb"))

@app.route("/")
def home():
    return "Cars MPG Prediction App Running"

@app.route("/predict", methods=["POST"])
def predict():
    hp = float(request.form['HP'])
    sp = float(request.form['SP'])
    vol = float(request.form['VOL'])

    prediction = model.predict([[vol, sp, hp]])

    return f"Predicted MPG: {prediction[0]:.2f}"

if __name__ == "__main__":
    # ONLY ONE RUN for Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
