from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("cars_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        vol = float(request.form['VOL'])
        sp = float(request.form['SP'])
        hp = float(request.form['HP'])
        pred = model.predict([[vol, sp, hp]])
        prediction = f"{pred[0]:.2f}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
