from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load("dt.joblib")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    data = request.get_json()
    features = np.array([
        data["sepal length (cm)"],
        data["sepal width (cm)"],
        data["petal length (cm)"],
        data["petal width (cm)"]
    ]).reshape(1, -1)

    # Predict using the trained model
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
