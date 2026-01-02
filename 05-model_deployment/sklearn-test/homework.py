try:
    import pickle

    with open('pipeline_v1.bin', 'rb') as f_in:
        pipeline = pickle.load(f_in)

    record = {
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0
    }

    proba = pipeline.predict_proba([record])[0, 1]
    print(f"Probability of conversion: {proba:.3f}")

except Exception as e:
    print(f"An error occurred: {e}")
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the pipeline
with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# Define the request schema
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Create FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(lead: Lead):
    record = lead.dict()
    proba = pipeline.predict_proba([record])[0, 1]
    return {"subscription_probability": round(proba, 3)}


import requests

url = "http://127.0.0.1:8000/predict"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=client)
print(response.json())