# app.py
import os
import sys

# Set the Python executable paths for PySpark
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import streamlit as st
import pandas as pd
import requests
import json
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, when, regexp_extract

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor (Spark)",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize Spark session and model
@st.cache_resource
def get_spark_session():
    """Initialize and return a Spark Session."""
    return (
        SparkSession.builder
        .appName("StreamlitSparkApp")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "1g")
        .master("local[*]")
        .getOrCreate()
    )

@st.cache_resource
def load_model(_spark, model_path: str):
    """Load the trained Spark ML model."""
    try:
        model = PipelineModel.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load Spark model from {model_path}. Error: {e}")
        return None

# Preprocessing function
def preprocess_passenger_data(spark_df):
    """Preprocess passenger data with feature engineering."""
    spark_df = spark_df.withColumn("Title", regexp_extract(col("Name"), " ([A-Za-z]+)\\.", 1))
    spark_df = spark_df.withColumn("Title", when(col("Title") == "", "Other").otherwise(col("Title")))
    spark_df = spark_df.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)
    spark_df = spark_df.withColumn("IsAlone", when(col("FamilySize") == 1, 1).otherwise(0))
    spark_df = spark_df.withColumn("Age_imputed", col("Age"))
    return spark_df

def main():
    # App title and description
    st.title("🚢 Titanic Survival Predictor (Spark Edition)")
    st.markdown("""
    This app uses a distributed **Spark MLlib** model to predict passenger survival 
    on the Titanic. Enter the passenger details below to get a prediction.
    """)
    
    # Initialize Spark and load model
    spark = get_spark_session()
    model_path = os.environ.get("MODEL_PATH", "model/spark_model")
    model = load_model(spark, model_path)
    
    if not model:
        st.error("Model could not be loaded. Please make sure the model is trained and available.")
        return
    
    # Sidebar for additional options
    with st.sidebar:
        st.header("Options")
        use_api = st.checkbox("Use API endpoint", value=False)
        api_url = st.text_input("API URL", "http://localhost:8000")
        
        st.header("About")
        st.info("""
        This app predicts survival on the Titanic using a Spark ML model.
        The model was trained on historical passenger data with features like
        class, age, gender, and fare.
        """)
    
    # User input section
    with st.form("prediction_form"):
        st.header("Enter Passenger Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pclass = st.selectbox("Ticket Class (Pclass):", [1, 2, 3], help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
            sex = st.radio("Sex:", ('male', 'female'), horizontal=True)
            age = st.slider("Age:", 0, 100, 30)
            embarked = st.selectbox("Port of Embarkation:", ('S', 'C', 'Q'), 
                                   help="S = Southampton, C = Cherbourg, Q = Queenstown")
        
        with col2:
            fare = st.slider("Fare:", 0, 512, 32)
            sibsp = st.slider("Siblings/Spouses Aboard (SibSp):", 0, 8, 0)
            parch = st.slider("Parents/Children Aboard (Parch):", 0, 6, 0)
            name = st.text_input("Passenger Name:", value="Cumings, Mrs. John Bradley (Florence Briggs Thayer)")
        
        submitted = st.form_submit_button("Predict Survival")
    
    if submitted:
        # Prepare input data
        passenger_data = {
            'Pclass': pclass,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Sex': sex,
            'Embarked': embarked,
            'Name': name
        }
        
        if use_api:
            # Use API endpoint for prediction
            try:
                response = requests.post(f"{api_url}/predict", json=passenger_data)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result['prediction']
                    probability = result['probability_survived']
                else:
                    st.error(f"API error: {response.status_code} - {response.text}")
                    return
            except Exception as e:
                st.error(f"Failed to call API: {e}")
                return
        else:
            # Use local model for prediction
            try:
                pandas_df = pd.DataFrame([passenger_data])
                spark_df = spark.createDataFrame(pandas_df)
                spark_df = preprocess_passenger_data(spark_df)
                prediction_df = model.transform(spark_df)
                result = prediction_df.select("prediction", "probability").first()
                
                prediction = int(result['prediction'])
                probability = float(result['probability'][1])
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return
        
        # Display results
        st.header("Prediction Result")
        prob_percent = probability * 100
        
        if prediction == 1:
            st.success(f"**This passenger would have likely survived!**")
            st.progress(probability)
            st.metric(label="Chance of Survival", value=f"{prob_percent:.2f}%")
            st.balloons()
        else:
            st.error(f"**This passenger would likely not have survived.**")
            st.progress(probability)
            st.metric(label="Chance of Survival", value=f"{prob_percent:.2f}%")
        
        # Show passenger details
        with st.expander("Passenger Details"):
            st.json(passenger_data)

if __name__ == '__main__':
    main()