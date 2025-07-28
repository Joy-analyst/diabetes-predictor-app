import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your trained XGBoost model (make sure this file exists)
model = joblib.load("xgb_diabetes_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data.get('features', None)

    if features is None or len(features) != 8:
        return jsonify({'error': 'Please provide 8 numeric features.'}), 400

    try:
        features = np.array(features, dtype=float).reshape(1, -1)
    except Exception:
        return jsonify({'error': 'Invalid feature values. Make sure all are numbers.'}), 400

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    result = "DIABETIC" if prediction == 1 else "NON-DIABETIC"

    return jsonify({
        "prediction": result,
        "probability": round(float(proba), 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
