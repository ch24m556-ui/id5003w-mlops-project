# src/train.py

import argparse
import mlflow
import json

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Import the MLflow client to interact with the registry
from mlflow.tracking import MlflowClient

def main(train_data_path, model_output_path):
    spark = SparkSession.builder.appName("SparkMLlibTraining").getOrCreate()

    with mlflow.start_run() as run:
        print("✅ MLflow Run Started.")
        df = spark.read.parquet(train_data_path)
        print("   - Data loaded into Spark DataFrame.")

        # --- Feature Engineering Pipeline ---
        categorical_cols = ['Sex', 'Embarked']
        numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        stages = []
        for col in categorical_cols:
            indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index")
            encoder = OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec")
            stages += [indexer, encoder]
        feature_cols = numerical_cols + [f"{col}_vec" for col in categorical_cols]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        stages += [assembler]
        
        # --- Model Definition ---
        max_depth = 5
        mlflow.log_param("max_depth", max_depth)
        dt = DecisionTreeClassifier(labelCol="Survived", featuresCol="features", maxDepth=max_depth)
        stages += [dt]
        
        # --- Training ---
        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(df)
        print("   - Model training complete.")

        # --- Evaluation ---
        predictions = model.transform(df)
        evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")
        accuracy = evaluator_accuracy.evaluate(predictions)
        f1 = evaluator_f1.evaluate(predictions)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        print(f"   - Logging metrics: Accuracy={accuracy:.4f}, F1 Score={f1:.4f}")

        # --- Save DVC Outputs ---
        metrics = {"accuracy": accuracy, "f1_score": f1}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print("   - Metrics saved to metrics.json")
        model.write().overwrite().save(model_output_path)
        print(f"   - Spark model saved to {model_output_path}")

        # --- MLflow Model Logging and Registration (NEW) ---
        mlflow.spark.log_model(model, "spark-model", registered_model_name="TitanicSparkModel")
        print("   - Model logged to MLflow.")

        # Initialize MLflow client and get the model URI
        client = MlflowClient()
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/spark-model"
        
        # Register the model and get its version
        model_version = mlflow.register_model(model_uri, "TitanicSparkModel")
        print(f"   - Registered model 'TitanicSparkModel', version {model_version.version}.")

        # Transition the new model version to the "Staging" stage
        client.transition_model_version_stage(
            name="TitanicSparkModel",
            version=model_version.version,
            stage="Staging"
        )
        print(f"   - Transitioned model version {model_version.version} to 'Staging'.")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a distributed model and register it with MLflow")
    parser.add_argument("--train_data", required=True, help="Path to processed train data")
    parser.add_argument("--model_out", required=True, help="Path to save the final Spark model")
    args = parser.parse_args()
    main(args.train_data, args.model_out)