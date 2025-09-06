# api.py
import os
import sys

# Set the Python executable paths for PySpark
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import pandas as pd
import logging
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, when, regexp_extract
from typing import List, Optional
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI app
app = FastAPI(
    title="🚢 Titanic Survival Prediction API (Spark)",
    description="An API for predicting passenger survival on the Titanic using a Spark MLlib model.",
    version="2.0",
)

# Global variables for Spark session and model
spark = None
model = None

# Pydantic models
class PassengerFeatures(BaseModel):
    Pclass: int = Field(..., description="Ticket class (1, 2, or 3)", ge=1, le=3)
    Age: float = Field(..., description="Age in years", ge=0, le=100)
    SibSp: int = Field(..., description="Number of siblings / spouses aboard", ge=0, le=10)
    Parch: int = Field(..., description="Number of parents / children aboard", ge=0, le=10)
    Fare: float = Field(..., description="Passenger fare", ge=0, le=600)
    Sex: str = Field(..., description="Sex of the passenger ('male' or 'female')")
    Embarked: str = Field(..., description="Port of Embarkation ('C', 'Q', or 'S')")
    Name: str = Field(..., description="Passenger's name")

class BatchPredictionRequest(BaseModel):
    passengers: List[PassengerFeatures]

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_survived: float
    passenger_details: dict

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

@app.on_event("startup")
async def startup_event():
    """Initialize Spark session and load model on startup."""
    global spark, model
    
    try:
        # Initialize Spark session
        spark = SparkSession.builder \
            .appName("TitanicPredictionAPI") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .master("local[*]") \
            .getOrCreate()
        
        logging.info(" Spark Session initialized.")
        
        # Load the trained model
        model_path = os.environ.get("MODEL_PATH", "model/spark_model")
        model = PipelineModel.load(model_path)
        logging.info(f" Spark model loaded successfully from {model_path}")
        
    except Exception as e:
        logging.error(f" Failed to initialize Spark or load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize service: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop Spark session on shutdown."""
    if spark:
        spark.stop()
        logging.info(" Spark Session stopped.")

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Titanic Survival Prediction API",
        "version": "2.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    if not spark or not model:
        raise HTTPException(status_code=503, detail="Spark or Model not available.")
    
    return {
        "status": "healthy", 
        "spark_running": True, 
        "model_loaded": True,
        "model_path": os.environ.get("MODEL_PATH", "model/spark_model")
    }

def preprocess_passenger_data(spark_df):
    """Preprocess passenger data with feature engineering."""
    # Extract title from name
    spark_df = spark_df.withColumn("Title", regexp_extract(col("Name"), " ([A-Za-z]+)\\.", 1))
    spark_df = spark_df.withColumn("Title", when(col("Title") == "", "Other").otherwise(col("Title")))
    
    # Add FamilySize and IsAlone features
    spark_df = spark_df.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)
    spark_df = spark_df.withColumn("IsAlone", when(col("FamilySize") == 1, 1).otherwise(0))
    
    # Use Age as Age_imputed (assuming no missing values in API input)
    spark_df = spark_df.withColumn("Age_imputed", col("Age"))
    
    return spark_df

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: PassengerFeatures):
    """Predict survival for a single passenger."""
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        # Convert to Spark DataFrame
        pandas_df = pd.DataFrame([features.dict()])
        spark_df = spark.createDataFrame(pandas_df)
        
        # Preprocess data
        spark_df = preprocess_passenger_data(spark_df)
        
        # Make prediction
        prediction_df = model.transform(spark_df)
        result = prediction_df.select("prediction", "probability").first()
        
        # Format response
        prediction = int(result['prediction'])
        probability = float(result['probability'][1])  # Probability of survival
        
        return PredictionResponse(
            prediction=prediction,
            prediction_label="Survived" if prediction == 1 else "Did not survive",
            probability_survived=round(probability, 4),
            passenger_details=features.dict()
        )
    
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """Predict survival for multiple passengers."""
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        # Convert to Spark DataFrame
        passenger_dicts = [p.dict() for p in request.passengers]
        pandas_df = pd.DataFrame(passenger_dicts)
        spark_df = spark.createDataFrame(pandas_df)
        
        # Preprocess data
        spark_df = preprocess_passenger_data(spark_df)
        
        # Make predictions
        prediction_df = model.transform(spark_df)
        
        # Collect results
        results = []
        for i, row in enumerate(prediction_df.collect()):
            prediction = int(row['prediction'])
            probability = float(row['probability'][1])
            
            results.append(PredictionResponse(
                prediction=prediction,
                prediction_label="Survived" if prediction == 1 else "Did not survive",
                probability_survived=round(probability, 4),
                passenger_details=passenger_dicts[i]
            ))
        
        return BatchPredictionResponse(predictions=results)
    
    except Exception as e:
        logging.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)