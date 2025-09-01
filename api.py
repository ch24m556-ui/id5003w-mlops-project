# api.py

import os
import time
import joblib
import pandas as pd
import logging
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request, Depends
# Import the modern 'field_validator'
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings

# For Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- 1. Settings Management ---
class Settings(BaseSettings):
    model_path: str = "model/model.joblib"
    app_name: str = "Titanic Survival Prediction API"
    class Config:
        env_file = ".env"
settings = Settings()

# --- 2. Logging and Rate Limiting Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
limiter = Limiter(key_func=get_remote_address)

# --- 3. FastAPI App Initialization ---
app = FastAPI(
    title=settings.app_name,
    description="An advanced MLOps API for predicting passenger survival.",
    version="2.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- 4. Middleware for Performance Monitoring ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logging.info(f"Request '{request.method} {request.url.path}' processed in {process_time:.4f}s")
    return response

# --- 5. Model Loading (Cached) ---
@lru_cache(maxsize=1)
def load_model(model_path: str):
    try:
        model = joblib.load(model_path)
        logging.info(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        raise

def get_model():
    try:
        return load_model(settings.model_path)
    except Exception:
        raise HTTPException(status_code=503, detail="Model could not be loaded.")

# --- 6. Pydantic Models (Using modern @field_validator) ---
class PassengerFeatures(BaseModel):
    Pclass: int = Field(..., description="Ticket class (1, 2, or 3)")
    Age: float = Field(..., description="Age in years")
    SibSp: int = Field(..., description="Number of siblings / spouses aboard")
    Parch: int = Field(..., description="Number of parents / children aboard")
    Fare: float = Field(..., description="Passenger fare")
    Sex_male: bool = Field(..., description="True if the passenger is male")
    Embarked_Q: bool = Field(..., description="True if embarked from Queenstown")
    Embarked_S: bool = Field(..., description="True if embarked from Southampton")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Pclass": 3, "Age": 22.0, "SibSp": 1, "Parch": 0, "Fare": 7.25,
                "Sex_male": True, "Embarked_Q": False, "Embarked_S": True,
            }
        }
    )

    # **THE FIX: Use @field_validator instead of @validator**
    @field_validator('Pclass')
    def pclass_must_be_valid(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError('Pclass must be 1, 2, or 3')
        return v
    
    @field_validator('Age', 'Fare', 'SibSp', 'Parch')
    def must_be_non_negative(cls, v: float):
        if v < 0:
            # No need for the 'info' object here as Pydantic provides a good default error
            raise ValueError('must be a non-negative number')
        return v

class PredictionResponse(BaseModel):
    prediction: int = Field(description="The survival prediction (1 for Survived, 0 for Not Survived)")
    prediction_label: str = Field(description="A human-readable label for the prediction")
    probability_survived: float = Field(description="The model's confidence in the survival prediction")

# --- 7. API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    return {"message": f"Welcome to the {settings.app_name}. Visit /docs for documentation."}

@app.get("/health", tags=["General"], summary="Check if the API is running")
def health_check(model=Depends(get_model)):
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"], summary="Predict passenger survival")
@limiter.limit("10/minute")
def predict(request: Request, features: PassengerFeatures, model=Depends(get_model)):
    df = pd.DataFrame([features.dict()])
    try:
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label="Survived" if int(prediction) == 1 else "Did not survive",
            probability_survived=round(probability[1], 4)
        )
    except Exception as e:
        logging.error(f"Prediction error for input {features.dict()}: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")