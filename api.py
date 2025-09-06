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

# --- 1. Application and Spark Session Management ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="🚢 Titanic Survival Prediction API (Spark)",
    description="An API for predicting passenger survival on the Titanic using a Spark MLlib model.",
    version="2.0",
)

spark = None
model = None

@app.on_event("startup")
def startup_event():
    """
    On startup, initialize a Spark Session and load the trained model.
    """
    global spark, model
    logging.info("Starting up API...")
    
    # Initialize a Spark Session for the API
    spark = SparkSession.builder \
        .appName("TitanicPredictionAPI") \
        .master("local[*]") \
        .getOrCreate()
    logging.info("✅ Spark Session initialized.")

    # Load the trained Spark ML Pipeline model
    model_path = "model/spark_model"
    try:
        model = PipelineModel.load(model_path)
        logging.info(f"Spark model loaded successfully from {model_path}")
    except Exception as e:
        logging.error(f" Failed to load Spark model: {e}")
        model = None

@app.on_event("shutdown")
def shutdown_event():
    """
    On shutdown, stop the Spark Session.
    """
    if spark:
        spark.stop()
        logging.info("Spark Session stopped.")

# --- 2. Pydantic Input Model ---
class PassengerFeatures(BaseModel):
    Pclass: int = Field(..., description="Ticket class (1, 2, or 3)")
    Age: float = Field(..., description="Age in years")
    SibSp: int = Field(..., description="Number of siblings / spouses aboard")
    Parch: int = Field(..., description="Number of parents / children aboard")
    Fare: float = Field(..., description="Passenger fare")
    Sex: str = Field(..., description="Sex of the passenger ('male' or 'female')")
    Embarked: str = Field(..., description="Port of Embarkation ('C', 'Q', or 'S')")
    Name: str = Field(..., description="Passenger's name")

# --- 3. API Endpoints ---
@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint."""
    if not spark or not model:
        raise HTTPException(status_code=503, detail="Spark or Model not available.")
    return {"status": "ok", "spark_running": True, "model_loaded": True}

@app.post("/predict", tags=["Prediction"])
def predict(features: PassengerFeatures):
    """
    Takes passenger features, converts them to a Spark DataFrame,
    and returns a survival prediction using the loaded Spark model.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    # Convert the Pydantic model to a Pandas DataFrame, then to a Spark DataFrame
    passenger_dict = features.dict()
    pandas_df = pd.DataFrame([passenger_dict])
    spark_df = spark.createDataFrame(pandas_df)
    
    # Perform the same feature engineering as in data_processing.py
    spark_df = spark_df.withColumn("Title", regexp_extract(col("Name"), " ([A-Za-z]+)\\.", 1))
    spark_df = spark_df.withColumn("Title", when(col("Title") == "", "Other").otherwise(col("Title")))
    
    # Add FamilySize and IsAlone features
    spark_df = spark_df.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)
    spark_df = spark_df.withColumn("IsAlone", when(col("FamilySize") == 1, 1).otherwise(0))
    
    # Use Age as Age_imputed (assuming no missing values in API input)
    spark_df = spark_df.withColumn("Age_imputed", col("Age"))
    
    # Use the model to make a prediction
    prediction_df = model.transform(spark_df)
    
    # Extract the results
    result = prediction_df.select("prediction", "probability").first()
    prediction = int(result['prediction'])
    probability = result['probability'][1]  # Probability of survival (class 1)

    return {
        "prediction": prediction,
        "prediction_label": "Survived" if prediction == 1 else "Did not survive",
        "probability_survived": round(probability, 4)
    }