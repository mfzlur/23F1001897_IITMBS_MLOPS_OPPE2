from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import pandas as pd
import time
import logging
import json
import uvicorn
import uuid
from datetime import datetime

# Setup Structured Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops-logger")

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

# Load model globally
try:
    model = joblib.load('model.joblib')
except FileNotFoundError:
    logger.error("Model file not found. Ensure 'model.joblib' is in the directory.")
    model = None

# --- Schemas ---
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

class BatchPredictionResponse(BaseModel):
    batch_id: str
    predictions: List[str]
    probabilities: List[float]
    processing_time: float

class FeedbackData(BaseModel):
    prediction_id: str
    actual_label: str = Field(..., description="The ground truth: 'yes' or 'no'")
    # Optional: Include original features if you don't store them by ID elsewhere
    features: Optional[HealthData] = None

# --- Middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # We log basic metrics for every call
    log_dict = {
        "path": request.url.path,
        "method": request.method,
        "status_code": response.status_code,
        "latency_seconds": round(process_time, 4)
    }
    logger.info(json.dumps(log_dict))
    return response

# --- Helper Functions ---
def log_feedback(data: FeedbackData):
    # In production, this would write to a DB or Feature Store
    log_payload = {
        "event": "feedback_received",
        "prediction_id": data.prediction_id,
        "actual_label": data.actual_label,
        "timestamp": datetime.utcnow().isoformat()
    }
    logger.info(json.dumps(log_payload))

# --- Endpoints ---

@app.get("/health")
def health():
    model_status = "loaded" if model else "failed"
    return {"status": "healthy", "model_status": model_status}

@app.get("/model/metadata")
def get_model_metadata():
    """
    Exposes model details for documentation and frontend integration.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Extracting basic info from sklearn pipeline if possible
    model_type = type(model.named_steps['classifier']).__name__ if hasattr(model, 'named_steps') else str(type(model))
    
    return {
        "model_name": "Heart Disease Classifier",
        "model_type": model_type,
        "version": "v1.0.0",  # Ideally read from a config file
        "required_features": list(HealthData.schema()['properties'].keys()),
        "classes": ["no", "yes"]
    }

@app.post("/predict", response_model=dict)
def predict(data: HealthData):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Generate a unique ID for this prediction event for traceability
    prediction_id = str(uuid.uuid4())
    
    df_input = pd.DataFrame([data.dict()])
    
    prediction = model.predict(df_input)
    probability = model.predict_proba(df_input).max()
    
    result = {
        "prediction_id": prediction_id,
        "prediction": "yes" if prediction[0] == 1 else "no",
        "probability": round(float(probability), 4)
    }
    
    # Log with ID so we can match feedback later
    logger.info(json.dumps({
        "event": "prediction", 
        "prediction_id": prediction_id, 
        "input": data.dict(), 
        "output": result
    }))
    
    return result

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(data: List[HealthData]):
    """
    Efficiently process multiple records in a single request.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    batch_id = str(uuid.uuid4())
    
    # 1. Convert list of objects to DataFrame (Vectorization)
    df_input = pd.DataFrame([item.dict() for item in data])
    
    # 2. Batch Inference
    predictions = model.predict(df_input)
    probs = model.predict_proba(df_input).max(axis=1)
    
    # 3. Map numeric predictions to labels
    pred_labels = ["yes" if p == 1 else "no" for p in predictions]
    
    # 4. Log summary
    logger.info(json.dumps({
        "event": "batch_prediction",
        "batch_id": batch_id,
        "count": len(data)
    }))

    return {
        "batch_id": batch_id,
        "predictions": pred_labels,
        "probabilities": [round(float(p), 4) for p in probs],
        "processing_time": round(time.time() - start, 4)
    }

@app.post("/feedback")
def submit_feedback(data: FeedbackData, background_tasks: BackgroundTasks):
    """
    Endpoint to receive ground truth labels for monitoring model drift.
    Uses BackgroundTasks to avoid blocking the response.
    """
    # Verify we actually received a valid label
    if data.actual_label not in ['yes', 'no']:
        raise HTTPException(status_code=400, detail="Invalid label. Must be 'yes' or 'no'")
        
    # Offload the logging/DB write to background task
    background_tasks.add_task(log_feedback, data)
    
    return {"status": "feedback_received", "id": data.prediction_id}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)