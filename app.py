from fastapi import FastAPI
import joblib
import pandas as pd
import json

app = FastAPI()

model = joblib.load("model.pkl")

with open("features.json") as f:
    feature_names = json.load(f)

@app.post("/predict")
def predict(data: dict):
    row = {feat: data.get(feat, 0) for feat in feature_names}
    df = pd.DataFrame([row])
    print(df)  # DEBUG
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}
