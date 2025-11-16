import pickle
from flask import Flask, request, jsonify

# Load model and vectorizer
model_file = 'heart_model.pkl'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

# Initialize Flask app
app = Flask('heart-disease-predictor')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive patient data as JSON
    patient = request.get_json()

    # Transform input using DictVectorizer
    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0, 1]
    risk = y_pred >= 0.5

    # Format result
    result = {
        'heart_disease_probability': float(y_pred),
        'at_risk': bool(risk)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)