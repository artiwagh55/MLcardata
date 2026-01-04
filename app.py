from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("cars_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")  # <-- render HTML template

@app.route("/predict", methods=["POST"])
def predict():
    vol = float(request.form['VOL'])
    sp = float(request.form['SP'])
    hp = float(request.form['HP'])

    prediction = model.predict([[vol, sp, hp]])

    # Render template with prediction
    return render_template("index.html", prediction=round(prediction[0], 2))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
