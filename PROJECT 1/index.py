import os
import requests
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Function to download the model from Google Drive
def download_model_from_drive():
    file_id = "11etsrHkbQzwXkSILUng4hS0FyoTHvFIv"  
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(download_url)
    with open("dt.joblib", "wb") as f:
        f.write(response.content)

# Check if the model exists; if not, download it
if not os.path.exists("dt.joblib"):
    print("Downloading model from Google Drive...")
    download_model_from_drive()

# Load the model
try:
    model = joblib.load("dt.joblib")
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Make prediction
        input_features = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_features)[0]
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
