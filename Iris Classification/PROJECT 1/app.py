from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

try:
    model = pickle.load(open("dt.pkl", "rb"))  
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file 'dt.pkl' not found. Ensure the file is in the correct directory.")
    model = None  

@app.route("/", methods=["GET"])
def home():
    """Serve the front-end HTML."""
    return render_template("index.html")  

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            raise Exception("Model not loaded. Please upload 'dt.pkl' to the server.")
      
        data = request.get_json()
        if not data:
            raise ValueError("No data provided. Ensure the request body contains valid JSON.")
        print("Received data:", data)

        feature_mapping = {
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width"
        }
       
        processed_data = {feature_mapping[key]: value for key, value in data.items()}
        
        df = pd.DataFrame([processed_data])
        print("Processed data for prediction:", df)
       
        prediction = model.predict(df)
        prediction_result = prediction.tolist()[0] 

        return jsonify({"prediction": prediction_result})

    except Exception as e:
        print("Error:", str(e))  
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
