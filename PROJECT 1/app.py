# from flask import Flask, request, jsonify, render_template
# import pickle
# from flask_cors import CORS
# import pandas as pd

# app = Flask(__name__)
# CORS(app)

# # Load your trained model
# model = pickle.load(open("dt.pkl", "rb"))  # Replace with your model file

# @app.route("/", methods=["GET"])
# def home():
#     return render_template("index.html")  # Serve the front-end HTML

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Log received JSON
#         data = request.get_json()
#         print("Received data:", data)

#         # Define a mapping of incoming feature names to the ones used by the model
#         feature_mapping = {
#             "sepal length (cm)": "sepal_length",
#             "sepal width (cm)": "sepal_width",
#             "petal length (cm)": "petal_length",
#             "petal width (cm)": "petal_width"
#         }
        
#         # Preprocess the data to match the expected feature names
#         processed_data = {feature_mapping[key]: value for key, value in data.items()}

#         # Convert the processed data to a DataFrame
#         df = pd.DataFrame([processed_data])
        
#         # Predict using the trained model
#         prediction = model.predict(df)
        
#         # If prediction is an array, convert it to a list before sending the response
#         prediction_result = prediction.tolist()[0]  # For single prediction value
        
#         # Send the prediction as a response
#         return jsonify({"prediction": prediction_result})
    
#     except Exception as e:
#         print("Error:", str(e))  # Log the error for debugging
#         return jsonify({"error": str(e)}), 400

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your trained model
model = pickle.load(open("dt.pkl", "rb"))  # Replace with your model file

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Serve the front-end HTML

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Log received JSON
        data = request.get_json()
        print("Received data:", data)

        # Define a mapping of incoming feature names to the ones used by the model
        feature_mapping = {
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width"
        }
        
        # Preprocess the data to match the expected feature names
        processed_data = [data.get("sepal length (cm)", 0), 
                          data.get("sepal width (cm)", 0), 
                          data.get("petal length (cm)", 0), 
                          data.get("petal width (cm)", 0)]

        # Predict using the trained model
        prediction = model.predict([processed_data])
        
        # If prediction is an array, convert it to a list before sending the response
        prediction_result = prediction[0]  # For single prediction value
        
        # Send the prediction as a response
        return jsonify({"prediction": prediction_result})
    
    except Exception as e:
        print("Error:", str(e))  # Log the error for debugging
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
