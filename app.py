# app.py

import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

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

# THE FIX IS HERE: The 'spark' argument is now '_spark'
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
    # The function call here does not need to change
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

            with col2:
                fare = st.slider("Fare:", 0, 512, 32)
                sibsp = st.slider("Siblings/Spouses Aboard (SibSp):", 0, 8, 0)
                parch = st.slider("Parents/Children Aboard (Parch):", 0, 6, 0)

            submitted = st.form_submit_button("Predict Survival")

        if submitted:
            # --- Prediction Logic ---
            user_input = {
                'Pclass': pclass, 'Age': age, 'SibSp': sibsp, 
                'Parch': parch, 'Fare': fare, 'Sex': sex, 'Embarked': embarked
            }
            
            pandas_df = pd.DataFrame([user_input])
            spark_df = spark.createDataFrame(pandas_df)
            
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