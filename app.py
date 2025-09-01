# app.py

import streamlit as st
import pandas as pd
import joblib
import os

# --- Helper Functions ---

@st.cache_resource
def load_model(model_path: str):
    """
    Load the trained model from the specified path using joblib.
    The @st.cache_resource decorator caches the model for performance.
    """
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure the DVC pipeline has been run.")
        return None

def preprocess_input(user_data: dict) -> pd.DataFrame:
    """
    Takes user input from the Streamlit interface and transforms it into the
    one-hot encoded format that the trained model expects.

    :param user_data: A dictionary containing the user's selections.
    :return: A Pandas DataFrame ready for prediction.
    """
    # Create a DataFrame from the user's input
    df = pd.DataFrame([user_data])

    # --- Feature Engineering to Match Model Training ---
    # 1. Pclass: Already in the correct numerical format.
    
    # 2. Sex: Convert to the one-hot encoded 'Sex_male' column.
    df['Sex_male'] = df['Sex'].apply(lambda x: 1 if x == 'Male' else 0)

    # 3. Embarked: Convert to one-hot encoded 'Embarked_Q' and 'Embarked_S' columns.
    df['Embarked_Q'] = df['Embarked'].apply(lambda x: 1 if x == 'Queenstown' else 0)
    df['Embarked_S'] = df['Embarked'].apply(lambda x: 1 if x == 'Southampton' else 0)

    # 4. Drop original categorical columns that have been encoded.
    df = df.drop(['Sex', 'Embarked'], axis=1)
    
    # 5. Ensure the column order is exactly the same as during training.
    # Our trained model expects this specific order.
    expected_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    df = df[expected_columns]
    
    return df

# --- Streamlit App Main Function ---

def main():
    # --- Page Configuration ---
    st.set_page_config(
        page_title="Titanic Survival Predictor",
        page_icon="🚢",
        layout="centered"
    )

    # --- App Title and Description ---
    st.title("🚢 Titanic Survival Predictor")
    st.markdown("This app predicts whether a passenger would have survived the Titanic disaster based on their details. The model was trained as part of an MLOps pipeline.")

    # --- Model Loading ---
    model_path = "model/model.joblib"
    model = load_model(model_path)

    if model:
        # --- User Input Section ---
        with st.form("prediction_form"):
            st.header("Enter Passenger Details")

            # Create columns for a cleaner layout
            col1, col2 = st.columns(2)
            
            with col1:
                pclass = st.selectbox("Ticket Class (Pclass):", [1, 2, 3])
                sex = st.radio("Sex:", ('Male', 'Female'), horizontal=True)
                age = st.slider("Age:", 0, 100, 30)
                embarked = st.selectbox("Port of Embarkation:", ('Southampton', 'Cherbourg', 'Queenstown'))

            with col2:
                fare = st.slider("Fare:", 0, 512, 32, help="Fare paid for the ticket in pounds.")
                sibsp = st.slider("Siblings/Spouses Aboard (SibSp):", 0, 8, 0)
                parch = st.slider("Parents/Children Aboard (Parch):", 0, 6, 0)

            # Form submission button
            submitted = st.form_submit_button("Predict Survival")

        if submitted:
            # --- Prediction Logic ---
            user_input = {
                'Pclass': pclass, 'Age': age, 'SibSp': sibsp, 
                'Parch': parch, 'Fare': fare, 'Sex': sex, 'Embarked': embarked
            }
            
            # Preprocess the input and make a prediction
            processed_df = preprocess_input(user_input)
            prediction_proba = model.predict_proba(processed_df)[0][1]
            prediction = (prediction_proba > 0.5)

            # --- Display Results ---
            st.header("Prediction Result")
            prob_percent = prediction_proba * 100

            if prediction:
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