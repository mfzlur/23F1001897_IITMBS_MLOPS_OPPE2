from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import time
import logging
import json
import uvicorn

# Setup Structured Logging for Observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops-logger")

app = FastAPI()
model = joblib.load('model.joblib')

class HealthData(BaseModel):
    age: int
    gender: str
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: int
    thal: int

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    log_dict = {
        "path": request.url.path,
        "method": request.method,
        "status_code": response.status_code,
        "latency_seconds": round(process_time, 4)
    }
    logger.info(json.dumps(log_dict))
    return response

@app.post("/predict")
def predict(data: HealthData):
    # Convert Pydantic model to DataFrame
    df_input = pd.DataFrame([data.dict()])
    
    # Prediction
    prediction = model.predict(df_input)
    probability = model.predict_proba(df_input).max()
    
    result = {
        "prediction": "yes" if prediction[0] == 1 else "no",
        "probability": round(float(probability), 4)
    }
    
    # Log specific prediction details for observability
    logger.info(json.dumps({"event": "prediction", "input": data.dict(), "output": result}))
    
    return result

@app.get("/health")
def health():
    return {"status": "healthy!! congrats your k8s is working properly"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)