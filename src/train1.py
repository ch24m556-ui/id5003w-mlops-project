# src/train.py

import argparse
import mlflow
import json
import os

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    RandomForestClassifier,
    LogisticRegression
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from mlflow.tracking import MlflowClient

def main(train_data_path, model_output_path):
    """
    Main function to train and tune multiple classification models, track experiments
    with MLflow, and register the best-performing model.

    This script performs the following steps:
    1.  Initializes a Spark session and sets up MLflow tracking.
    2.  Starts a parent MLflow run to encapsulate the entire model selection process.
    3.  Loads and splits the training data into training and test sets.
    4.  Defines a common feature engineering pipeline for all models.
    5.  Defines three separate models to compare: Decision Tree, Random Forest, and
        Logistic Regression, each with its own hyperparameter grid.
    6.  Iterates through each model:
        a. Starts a nested MLflow run for tuning that specific model.
        b. Uses Cross-Validation to find the best hyperparameters.
        c. Logs the best cross-validation F1 score to the nested run.
    7.  Compares the best F1 scores from all tuning runs to select the overall best model.
    8.  Evaluates the final, best model on the held-out test set.
    9.  Logs the final performance metrics (accuracy, F1 score) and the metrics.json
        file as an artifact to the parent MLflow run.
    10. Saves the best model pipeline artifact to the specified output path.
    11. Logs the model to MLflow and registers it in the MLflow Model Registry,
        transitioning it directly to the 'Staging' environment.
    """
    spark = SparkSession.builder.appName("SparkMLlibMultiModelTuning").getOrCreate()

    # Set the MLflow tracking URI to a local directory. This is robust for local and CI/CD runs.
    mlflow.set_tracking_uri("./mlruns")

    # Start a parent MLflow run to encompass the entire model comparison process.
    with mlflow.start_run(run_name="Model Comparison and Tuning") as parent_run:
        print("🚀 MLflow Parent Run Started for Model Comparison and Tuning.")
        df = spark.read.parquet(train_data_path)
        (training_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)
        print("   - Data loaded and split for training and validation.")

        # --- Define the Common Feature Engineering Pipeline ---
        categorical_cols = ['Sex', 'Embarked']
        numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        feature_stages = []

        for col in categorical_cols:
            # Using handleInvalid="skip" makes the pipeline robust to missing categorical values.
            indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="skip")
            encoder = OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec")
            feature_stages += [indexer, encoder]
        
        feature_cols = numerical_cols + [f"{col}_vec" for col in categorical_cols]
        # handleInvalid="skip" in VectorAssembler handles any nulls that may remain in numeric columns.
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
        feature_stages.append(assembler)
        print("   - Feature engineering pipeline defined.")

        # --- Define Models and Hyperparameter Grids to Compare ---
        dt = DecisionTreeClassifier(labelCol="Survived", featuresCol="features")
        rf = RandomForestClassifier(labelCol="Survived", featuresCol="features")
        lr = LogisticRegression(labelCol="Survived", featuresCol="features")

        # Grid for Decision Tree
        dt_param_grid = ParamGridBuilder() \
            .addGrid(dt.maxDepth, [3, 5, 7]) \
            .addGrid(dt.maxBins, [32, 40]) \
            .build()

        # Grid for Random Forest
        rf_param_grid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [10, 20, 30]) \
            .addGrid(rf.maxDepth, [3, 5, 7]) \
            .build()
        
        # Grid for Logistic Regression
        lr_param_grid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
            .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
            .build()
            
        models_to_tune = [
            ("DecisionTree", dt, dt_param_grid),
            ("RandomForest", rf, rf_param_grid),
            ("LogisticRegression", lr, lr_param_grid),
        ]

        evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")

        best_overall_model = None
        best_f1_score = -1.0
        
        # --- Iterate Through Models and Tune in Nested MLflow Runs ---
        print("\n--- Starting Hyperparameter Tuning for Each Model ---")
        for model_name, classifier, param_grid in models_to_tune:
            with mlflow.start_run(run_name=f"Tuning_{model_name}", nested=True) as child_run:
                print(f"\n   Tuning {model_name}...")
                
                # Combine feature stages with the current classifier to form a full pipeline
                pipeline = Pipeline(stages=feature_stages + [classifier])

                # Set up Cross-Validator for the current model
                cv = CrossValidator(estimator=pipeline,
                                    estimatorParamMaps=param_grid,
                                    evaluator=evaluator,
                                    numFolds=3,
                                    parallelism=2) # Increase parallelism based on available cores
                
                cv_model = cv.fit(training_data)
                
                # Find the best F1 score achieved during this model's tuning run
                current_best_f1 = max(cv_model.avgMetrics)
                print(f"   - Best cross-validated F1 score for {model_name}: {current_best_f1:.4f}")
                mlflow.log_metric("best_cv_f1_score", current_best_f1)
                
                # Check if this model is the best one found so far across all types
                if current_best_f1 > best_f1_score:
                    best_f1_score = current_best_f1
                    best_overall_model = cv_model.bestModel
                    print(f"   - ✨ New best overall model found: {model_name}")

        print("\n--- Tuning Complete. Evaluating Best Overall Model ---")

        # --- Evaluate and Save the Single Best Model ---
        predictions = best_overall_model.transform(test_data)
        accuracy = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
        f1_final = evaluator.evaluate(predictions)

        # Log final metrics to the parent run
        mlflow.log_metric("best_model_accuracy", accuracy)
        mlflow.log_metric("best_model_f1_score", f1_final)
        print(f"   - Final Metrics on Test Set: Accuracy={accuracy:.4f}, F1 Score={f1_final:.4f}")

        # Create and log metrics.json artifact
        metrics = {"accuracy": accuracy, "f1_score": f1_final}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact("metrics.json")
        
        best_overall_model.write().overwrite().save(model_output_path)
        print(f"   - ✅ Best Spark model saved to {model_output_path}")

        # --- Log and Register the Best Model in MLflow Model Registry ---
        model_name_registered = "TitanicSparkModel-MultiTuned"
        mlflow.spark.log_model(
            spark_model=best_overall_model,
            artifact_path="spark-model",
            registered_model_name=model_name_registered
        )
        
        # Transition the newly registered model version to the "Staging" stage
        client = MlflowClient()
        latest_version_info = client.get_latest_versions(model_name_registered, stages=["None"])
        if latest_version_info:
            latest_version = latest_version_info[0].version
            client.transition_model_version_stage(
                name=model_name_registered,
                version=latest_version,
                stage="Staging",
                archive_existing_versions=True
            )
            print(f"   - ✅ Registered and transitioned model version {latest_version} to 'Staging'.")
        else:
            print("   - ⚠️ Could not find a new model version to transition.")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune and train multiple distributed models with Spark MLlib")
    parser.add_argument("--train_data", required=True, help="Path to processed train data (Parquet format)")
    parser.add_argument("--model_out", required=True, help="Path to save the final best model artifact")
    args = parser.parse_args()
    main(args.train_data, args.model_out)
