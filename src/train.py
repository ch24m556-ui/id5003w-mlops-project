# src/train.py

import argparse
import mlflow
import json
import os

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from mlflow.tracking import MlflowClient

def main(train_data_path, model_output_path):
    spark = SparkSession.builder.appName("SparkMLlibTuning").getOrCreate()

    # Set the MLflow tracking URI to a safe, relative path to prevent CI/CD errors.
    mlflow.set_tracking_uri("./mlruns")

    # Use a nested run to group all tuning trials under one parent run
    with mlflow.start_run(run_name="Hyperparameter Tuning") as parent_run:
        print("(OK) MLflow Parent Run Started for Hyperparameter Tuning.")
        df = spark.read.parquet(train_data_path)
        (training_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)
        print("   - Data loaded and split for training and validation.")

        # --- Define the Feature Engineering Pipeline ---
        categorical_cols = ['Sex', 'Embarked']
        numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        stages = []

        for col in categorical_cols:
            # --- THIS IS THE DEFINITIVE FIX for NULL VALUES ---
            # Set handleInvalid="skip" to robustly handle any null values in
            # categorical columns by filtering out those rows. This prevents the
            # "StringIndexer encountered NULL value" error.
            indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="skip")
            encoder = OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec")
            stages += [indexer, encoder]
        
        feature_cols = numerical_cols + [f"{col}_vec" for col in categorical_cols]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
        stages += [assembler]

        # --- Define Models and Hyperparameter Grids ---
        dt = DecisionTreeClassifier(labelCol="Survived", featuresCol="features")
        lr = LogisticRegression(labelCol="Survived", featuresCol="features")

        param_grid = ParamGridBuilder() \
            .addGrid(dt.maxDepth, [3, 5, 7]) \
            .addGrid(lr.regParam, [0.01, 0.1]) \
            .build()
        
        evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")
        
        # We need a placeholder classifier here, CrossValidator will swap it out.
        pipeline_estimator = Pipeline(stages=stages + [dt])

        cv = CrossValidator(estimator=pipeline_estimator,
                            estimatorParamMaps=param_grid,
                            evaluator=evaluator,
                            numFolds=3,
                            parallelism=2)

        print("   - Starting Cross-Validation...")
        cv_model = cv.fit(training_data)
        print("   - Cross-Validation complete. Best model found.")

        # --- Evaluate and Save ---
        best_model = cv_model.bestModel
        predictions = best_model.transform(test_data)
        accuracy = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
        f1 = evaluator.evaluate(predictions)

        mlflow.log_metric("best_model_accuracy", accuracy)
        mlflow.log_metric("best_model_f1_score", f1)
        print(f"   - Logging best model metrics: Accuracy={accuracy:.4f}, F1 Score={f1:.4f}")

        metrics = {"accuracy": accuracy, "f1_score": f1}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        best_model.write().overwrite().save(model_output_path)
        print(f"   - Best Spark model saved to {model_output_path}")

        # --- Log and Register Model ---
        mlflow.spark.log_model(best_model, "spark-model", registered_model_name="TitanicSparkModel-Tuned")
        model_uri = f"runs:/{parent_run.info.run_id}/spark-model"
        model_version = mlflow.register_model(model_uri, "TitanicSparkModel-Tuned")
        client = MlflowClient()
        client.transition_model_version_stage(
            name="TitanicSparkModel-Tuned", version=model_version.version, stage="Staging"
        )
        print(f"   - Registered and transitioned model version {model_version.version} to 'Staging'.")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune and train a distributed model with Spark MLlib")
    parser.add_argument("--train_data", required=True, help="Path to processed train data")
    parser.add_argument("--model_out", required=True, help="Path to save the final best model")
    args = parser.parse_args()
    main(args.train_data, args.model_out)

