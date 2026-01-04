import os
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("cars_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")  # HTML form

@app.route("/predict", methods=["POST"])
def predict():
    hp = float(request.form['HP'])
    sp = float(request.form['SP'])
    vol = float(request.form['VOL'])
    
    prediction = model.predict([[vol, sp, hp]])
    return render_template("index.html", prediction=f"{prediction[0]:.2f}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
