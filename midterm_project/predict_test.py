#!/usr/bin/env python
# coding: utf-8

url = 'http://localhost:5000/predict'  # INFO:waitress:Serving on http://0.0.0.0:5000

patient_id = 'xyz-123'
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


import requests

response = requests.post(url, json=patient).json()

print(response)


if response['at_risk'] == True:
    print('sending health alert to %s' % patient_id)
else:
    print('no alert needed for %s' % patient_id)





