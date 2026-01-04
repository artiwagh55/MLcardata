from flask import Flask, request
import pickle
import numpy as np

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
    app.run(debug=True, host="127.0.0.1", port=8000)
