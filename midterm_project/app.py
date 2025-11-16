import requests

url = 'http://127.0.0.1:5000/predict'

patient = {
    "age": 54,
    "sex": 1,
    "cp": 0,
    "trestbps": 122,
    "chol": 286,
    "fbs": 0,
    "restecg": 0,
    "thalach": 116,
    "exang": 1,
    "oldpeak": 3.2,
    "slope": 1,
    "ca": 2,
    "thal": 2
}

response = requests.post(url, json=patient)
print(response.json())