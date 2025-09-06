# app.py

import os
import sys

# Set the Python executable paths for PySpark
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, when, regexp_extract

# --- Helper Functions ---
@st.cache_resource
def get_spark_session():
    """
    Initializes and returns a Spark Session.
    Using @st.cache_resource ensures the Spark Session is created only once.
    """
    return (
        SparkSession.builder
        .appName("StreamlitSparkApp")
        .master("local[*]")
        .getOrCreate()
    )

@st.cache_resource
def load_model(_spark, model_path: str):
    """
    Load the trained Spark ML Pipeline model.
    The leading underscore in '_spark' tells Streamlit to ignore this argument for caching.
    """
    try:
        # We still use the _spark object inside the function
        model = PipelineModel.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load Spark model from {model_path}. Error: {e}")
        return None

# --- Streamlit App Main Function ---
def main():
    # --- Page Configuration ---
    st.set_page_config(
        page_title="Titanic Survival Predictor (Spark)",
        page_icon="🚢",
        layout="centered"
    )

    # --- App Title and Description ---
    st.title("🚢 Titanic Survival Predictor (Spark Edition)")
    st.markdown("This app uses a distributed **Spark MLlib** model to predict passenger survival.")

    # --- Initialize Spark and Load Model ---
    spark = get_spark_session()
    model = load_model(spark, "model/spark_model")

    if model:
        # --- User Input Section ---
        with st.form("prediction_form"):
            st.header("Enter Passenger Details")

            col1, col2 = st.columns(2)
            
            with col1:
                pclass = st.selectbox("Ticket Class (Pclass):", [1, 2, 3])
                sex = st.radio("Sex:", ('male', 'female'), horizontal=True)
                age = st.slider("Age:", 0, 100, 30)
                embarked = st.selectbox("Port of Embarkation:", ('S', 'C', 'Q'))
                name = st.text_input("Passenger Name:", value="Cumings, Mrs. John Bradley (Florence Briggs Thayer)")

            with col2:
                fare = st.slider("Fare:", 0, 512, 32)
                sibsp = st.slider("Siblings/Spouses Aboard (SibSp):", 0, 8, 0)
                parch = st.slider("Parents/Children Aboard (Parch):", 0, 6, 0)

            submitted = st.form_submit_button("Predict Survival")

        if submitted:
            # --- Feature Engineering ---
            user_input = {
                'Pclass': pclass, 'Age': age, 'SibSp': sibsp, 
                'Parch': parch, 'Fare': fare, 'Sex': sex, 
                'Embarked': embarked, 'Name': name
            }
            
            pandas_df = pd.DataFrame([user_input])
            spark_df = spark.createDataFrame(pandas_df)
            
            # Perform the same feature engineering as in data_processing.py
            spark_df = spark_df.withColumn("Title", regexp_extract(col("Name"), " ([A-Za-z]+)\\.", 1))
            spark_df = spark_df.withColumn("Title", when(col("Title") == "", "Other").otherwise(col("Title")))
            
            # Add FamilySize and IsAlone features
            spark_df = spark_df.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)
            spark_df = spark_df.withColumn("IsAlone", when(col("FamilySize") == 1, 1).otherwise(0))
            
            # Use Age as Age_imputed (assuming no missing values in API input)
            spark_df = spark_df.withColumn("Age_imputed", col("Age"))
            
            # --- Prediction Logic ---
            prediction_df = model.transform(spark_df)
            result = prediction_df.select("prediction", "probability").first()
            
            prediction = int(result['prediction'])
            prediction_proba = result['probability'][1]

            # --- Display Results ---
            st.header("Prediction Result")
            prob_percent = prediction_proba * 100

            if prediction == 1:
                st.success(f"**You would have likely survived!**")
                st.progress(int(prob_percent))
                st.metric(label="Chance of Survival", value=f"{prob_percent:.2f}%")
                st.balloons()
            else:
                st.error(f"**You would likely not have survived.**")
                st.progress(int(prob_percent))
                st.metric(label="Chance of Survival", value=f"{prob_percent:.2f}%")

# --- Run the App ---
if __name__ == '__main__':
    main()