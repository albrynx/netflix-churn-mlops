# Save this as main.py
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('churn_model_balanced.pkl')
encoders = joblib.load('label_encoders.pkl')

@app.post("/predict_churn")
def predict(data: dict):
    # Convert input dict to DataFrame
    df_input = pd.DataFrame([data])
    
    # Process categorical inputs using saved encoders
    for col, le in encoders.items():
        df_input[col] = le.transform(df_input[col])
        
    prediction = model.predict(df_input)
    probability = model.predict_proba(df_input)[:, 1]
    
    return {
        "churn_prediction": "Yes" if prediction[0] == 1 else "No",
        "churn_probability": round(float(probability[0]), 2)
    }