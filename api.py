# api.py

import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import PySpark and related libraries
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

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
        logging.info(f"✅ Spark model loaded successfully from {model_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load Spark model: {e}")
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
    pandas_df = pd.DataFrame([features.dict()])
    spark_df = spark.createDataFrame(pandas_df)
    
    # Use the model to make a prediction (.transform is the Spark equivalent of .predict)
    prediction_df = model.transform(spark_df)
    
    # Extract the results from the Spark DataFrame
    # The result is a Row object, so we access its fields
    result = prediction_df.select("prediction", "probability").first()
    prediction = int(result['prediction'])
    probability = result['probability'][1]  # Probability of survival (class 1)

    return {
        "prediction": prediction,
        "prediction_label": "Survived" if prediction == 1 else "Did not survive",
        "probability_survived": round(probability, 4)
    }