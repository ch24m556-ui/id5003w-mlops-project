# src/train.py

import argparse
import mlflow
import json

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
# Import multiple classifiers
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Import tools for hyperparameter tuning
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from mlflow.tracking import MlflowClient

def main(train_data_path, model_output_path):
    spark = SparkSession.builder.appName("SparkMLlibTuning").getOrCreate()

    # Use a nested run to group all tuning trials under one parent run
    with mlflow.start_run(run_name="Hyperparameter Tuning") as parent_run:
        print("✅ MLflow Parent Run Started for Hyperparameter Tuning.")
        df = spark.read.parquet(train_data_path)
        # Split data for cross-validation
        (training_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)
        print("   - Data loaded and split for training and validation.")

        # --- 1. Define the Feature Engineering Pipeline ---
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

        # --- 2. Define Models and Hyperparameter Grids ---
        # We will test two different algorithms
        dt = DecisionTreeClassifier(labelCol="Survived", featuresCol="features")
        lr = LogisticRegression(labelCol="Survived", featuresCol="features")

        # Create a parameter grid for Decision Tree
        dt_param_grid = ParamGridBuilder() \
            .addGrid(dt.maxDepth, [3, 5, 7]) \
            .addGrid(dt.maxBins, [32, 40]) \
            .build()

        # Create a parameter grid for Logistic Regression
        lr_param_grid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
            .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
            .build()
        
        # Combine the parameter grids
        param_grid = dt_param_grid + lr_param_grid
        print("   - Defined models and hyperparameter grids for tuning.")

        # --- 3. Set Up the Cross-Validator ---
        # The CrossValidator will test all models and parameters
        evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")

        # The feature pipeline stages are combined with the classifier
        pipeline = Pipeline(stages=stages + [dt]) # Placeholder for classifier
        
        cv = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=param_grid,
                            evaluator=evaluator,
                            numFolds=3, # Use 3-fold cross-validation
                            parallelism=2) # Run two trials in parallel

        print("   - Starting Cross-Validation...")
        # This will automatically trigger MLflow to log child runs for each trial
        cv_model = cv.fit(training_data)
        print("   - Cross-Validation complete. Best model found.")

        # --- 4. Evaluate the Best Model and Log Metrics ---
        best_model = cv_model.bestModel
        predictions = best_model.transform(test_data)
        
        accuracy = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
        f1 = evaluator.evaluate(predictions)

        # Log metrics to the parent MLflow run
        mlflow.log_metric("best_model_accuracy", accuracy)
        mlflow.log_metric("best_model_f1_score", f1)
        print(f"   - Logging best model metrics: Accuracy={accuracy:.4f}, F1 Score={f1:.4f}")

        # --- 5. Save Outputs for DVC ---
        metrics = {"accuracy": accuracy, "f1_score": f1}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print("   - Metrics saved to metrics.json")

        best_model.write().overwrite().save(model_output_path)
        print(f"   - Best Spark model saved to {model_output_path}")

        # --- 6. Log and Register the Best Model ---
        mlflow.spark.log_model(best_model, "spark-model", registered_model_name="TitanicSparkModel-Tuned")
        print("   - Best model logged to MLflow.")
        
        client = MlflowClient()
        model_uri = f"runs:/{parent_run.info.run_id}/spark-model"
        model_version = mlflow.register_model(model_uri, "TitanicSparkModel-Tuned")
        print(f"   - Registered model 'TitanicSparkModel-Tuned', version {model_version.version}.")
        client.transition_model_version_stage(
            name="TitanicSparkModel-Tuned", version=model_version.version, stage="Staging"
        )
        print(f"   - Transitioned model version {model_version.version} to 'Staging'.")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune and train a distributed model with Spark MLlib")
    parser.add_argument("--train_data", required=True, help="Path to processed train data")
    parser.add_argument("--model_out", required=True, help="Path to save the final best model")
    args = parser.parse_args()
    main(args.train_data, args.model_out)