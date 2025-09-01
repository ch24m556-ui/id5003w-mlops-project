# src/train.py

import argparse
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import joblib # For saving the model
import json   # For saving metrics
import os     # For creating directories

def main(train_data_path, model_output_path):
    """
    Main function to train a model, logging with MLflow and saving artifacts for DVC.

    :param train_data_path: Path to the processed training data.
    :param model_output_path: Path to save the trained model file.
    """
    with mlflow.start_run():
        print("✅ MLflow Run Started.")

        df = pd.read_parquet(train_data_path)
        df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
        X = df.drop('Survived', axis=1)
        y = df['Survived']

        max_depth = 5
        random_state = 42

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        # Note: For a real project, we'd use a train/validation split.
        # Here, we train on all processed data for simplicity in the DVC pipeline.
        model.fit(X, y)
        print("   - Model training complete.")

        # Make predictions to calculate metrics
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        print(f"   - Logging metrics: Accuracy={accuracy:.4f}, F1 Score={f1:.4f}")

        # --- Saving outputs for DVC ---
        # Save metrics to a JSON file
        metrics = {"accuracy": accuracy, "f1_score": f1}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print("   - Metrics saved to metrics.json")

        # Create directory for model if it doesn't exist
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        # Save the trained model
        joblib.dump(model, model_output_path)
        print(f"   - Model saved to {model_output_path}")

        # Log the DVC-tracked model to MLflow as well for good measure
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with MLflow and DVC")
    parser.add_argument("--train_data", required=True, help="Path to processed train data")
    parser.add_argument("--model_out", required=True, help="Path to save the final model")
    args = parser.parse_args()
    main(args.train_data, args.model_out)