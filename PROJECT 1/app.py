from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
from flask_cors import CORS
import pandas as pd
import os
import urllib.request

app = Flask(__name__)
CORS(app)

# Google Drive direct download link for your model
MODEL_URL = "https://drive.google.com/uc?export=download&id=1G-l98KCOcgPBNjmahkAUaIwr3-cspT_i"  # Replace with your link
MODEL_PATH = "/tmp/dt.pkl"  # Temporary directory for serverless functions

# Function to download the model if not already downloaded
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")

# Load the trained model
try:
    download_model()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None  # Set to None to handle gracefully in prediction route

@app.route("/favicon.ico", methods=["GET"])
def favicon():
    """Serve a placeholder favicon to avoid 500 errors for /favicon.ico requests."""
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
        as_attachment=False
    )

@app.route("/", methods=["GET"])
def home():
    """Serve the front-end HTML."""
    return render_template("index.html")  # Ensure 'index.html' exists in 'templates' folder

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    try:
        # Check if the model is loaded
        if model is None:
            raise Exception("Model not loaded. Please upload 'dt.pkl' to the server.")

        # Log received JSON
        data = request.get_json()
        if not data:
            raise ValueError("No data provided. Ensure the request body contains valid JSON.")
        print("Received data:", data)

        # Define a mapping of incoming feature names to the ones used by the model
        feature_mapping = {
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width"
        }

        # Preprocess the data to match the expected feature names
        processed_data = {feature_mapping[key]: value for key, value in data.items()}

        # Convert the processed data to a DataFrame
        df = pd.DataFrame([processed_data])
        print("Processed data for prediction:", df)

        # Predict using the trained model
        prediction = model.predict(df)

        # If prediction is an array, convert it to a list before sending the response
        prediction_result = prediction.tolist()[0]  # For single prediction value

        # Send the prediction as a response
        return jsonify({"prediction": prediction_result})

    except Exception as e:
        print("Error:", str(e))  # Log the error for debugging
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
