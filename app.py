from fastapi import FastAPI
import joblib
import pandas as pd
import json

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

# Load feature names
with open("features.json") as f:
    feature_names = json.load(f)

@app.post("/predict")
def predict(data: dict):
    # Build row in correct order
    row = {feat: data.get(feat, 0) for feat in feature_names}
    df = pd.DataFrame([row])
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}
