"""
Flask API + web page for the Calories Burn Prediction project.

  GET  /              -> the web page (/index.html)
  GET  /api/meta      -> input ranges and test metrics for the UI
  POST /api/predict   -> returns the predicted calories

Run: python app.py
"""

import os
import json
import joblib
import pandas as pd

from flask import Flask, jsonify, request
from flask_cors import CORS


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


app = Flask(
    __name__,
    static_folder='.',
    static_url_path=''
)

CORS(app)


# Load the objects saved by train_model.py
artifacts = joblib.load(
    os.path.join(BASE_DIR, 'model', 'calories_model.joblib')
)

encoder = artifacts['encoder']
scaler = artifacts['scaler']
model = artifacts['model']


with open(
    os.path.join(BASE_DIR, 'model', 'metadata.json'),
    encoding='utf-8'
) as f:
    metadata = json.load(f)


# Exact model inputs
NUMERIC_FEATURES = [
    'Age',
    'Height',
    'Weight',
    'Duration',
    'Heart_Rate',
    'Body_Temp'
]


@app.route('/')
def home():
    return app.send_static_file('index.html')


@app.route('/api/meta')
def meta():
    return jsonify(metadata)


@app.route('/api/predict', methods=['POST'])
def predict():

    data = request.get_json(silent=True) or {}

    # Basic validation
    values = {}

    for name in NUMERIC_FEATURES:

        try:
            value = float(data[name])

        except (KeyError, TypeError, ValueError):
            return jsonify(
                error=f'{name} is required and must be a number.'
            ), 400

        values[name] = value


    # Validate gender
    gender = data.get('Gender')

    if gender not in metadata['genders']:
        return jsonify(
            error='Gender must be one of: '
                  + ', '.join(metadata['genders']) + '.'
        ), 400


    # --------------------------------
    # Existing ML pipeline
    # Encoder -> Scaler -> Model
    # --------------------------------

    # Create DataFrame for numeric features
    x_new = pd.DataFrame([values])


    # Encode Gender
    gender_encoded = encoder.transform(
        pd.DataFrame({'Gender': [gender]})
    )


    gender_df = pd.DataFrame(
        gender_encoded,
        columns=['Gender_female', 'Gender_male']
    )


    # Combine numeric + encoded features
    x_new = pd.concat(
        [x_new, gender_df],
        axis=1
    )


    # Make sure feature names and order
    # match the features used during training
    x_new = x_new[scaler.feature_names_in_]


    # Scale the input
    x_new = scaler.transform(x_new)


    # Predict calories
    calories = model.predict(x_new)[0]


    return jsonify(
        calories=round(float(calories), 1)
    )


if __name__ == '__main__':
    app.run(debug=True)