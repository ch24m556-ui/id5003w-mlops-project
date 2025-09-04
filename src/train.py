# src/train.py

# --- Environment Configuration for PySpark ---
import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# --- End of Configuration ---

import argparse
import mlflow
import json
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    RandomForestClassifier,
    LogisticRegression,
    GBTClassifier,
    DecisionTreeClassificationModel,
    RandomForestClassificationModel
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from mlflow.tracking import MlflowClient

# --- Helper Functions for Artifacts ---
def log_feature_importance(spark_model, predictions, artifact_path="feature_importance.png"):
    """Extracts and logs a feature importance plot for tree-based models."""
    if not isinstance(spark_model.stages[-1], (DecisionTreeClassificationModel, RandomForestClassificationModel)):
        print("   - Skipping feature importance: model is not a tree-based type.")
        return
    
    # Extract feature names directly from the fitted VectorAssembler stage metadata
    assembler_stage = next(s for s in spark_model.stages if isinstance(s, VectorAssembler))
    feature_attrs = predictions.schema[assembler_stage.getOutputCol()].metadata["ml_attr"]["attrs"]
    feature_names = []
    if "numeric" in feature_attrs:
        feature_names.extend([attr["name"] for attr in feature_attrs["numeric"]])
    if "binary" in feature_attrs:
        feature_names.extend([attr["name"] for attr in feature_attrs["binary"]])

    model = spark_model.stages[-1]
    importances = model.featureImportances.toArray()

    # Ensure lengths match before creating DataFrame
    if len(feature_names) != len(importances):
        print(f"   - ⚠️ Warning: Mismatch between feature names ({len(feature_names)}) and importances ({len(importances)}). Skipping plot.")
        return

    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(artifact_path)
    plt.close()
    mlflow.log_artifact(artifact_path)
    print(f"   - ✅ Logged feature importance plot to {artifact_path}")

def log_confusion_matrix(predictions, artifact_path="confusion_matrix.png"):
    """Generates and logs a confusion matrix plot."""
    preds_and_labels = predictions.select(['prediction', 'Survived']).withColumn('label', col('Survived').cast('float')).select(['prediction', 'label']).rdd.map(tuple)
    metrics = MulticlassMetrics(preds_and_labels)
    confusion_matrix = metrics.confusionMatrix().toArray()

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix'); plt.ylabel('Actual Class'); plt.xlabel('Predicted Class')
    plt.savefig(artifact_path)
    plt.close()
    mlflow.log_artifact(artifact_path)
    print(f"   - ✅ Logged confusion matrix plot to {artifact_path}")


# --- Main Training Logic ---
def main(train_data_path, model_output_path, profile_name):
    """
    Main function to train models on pre-processed, feature-engineered data.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    spark_configs = params["spark_configs"][profile_name]

    spark = SparkSession.builder.appName("SimplifiedSparkMLTraining").config(**spark_configs).getOrCreate()
    print(f"✅ Spark Session initialized with '{profile_name}' profile: {spark_configs}")

    mlflow.set_tracking_uri("./mlruns")
    with mlflow.start_run(run_name=f"Training with {profile_name} profile") as parent_run:
        mlflow.log_params(spark_configs)
        mlflow.log_param("experiment_profile", profile_name)
        print(f"🚀 MLflow Run Started for profile: {profile_name}.")

        df = spark.read.parquet(train_data_path)
        (training_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)
        print("   - Pre-processed data loaded and split for training and validation.")

        # --- Define the Simplified ML Pipeline ---
        # The data is already feature-engineered. We just need to encode and assemble.
        print("\n--- Defining Simplified Modeling Pipeline ---")

        # 1. Handle categorical columns created during preprocessing
        categorical_cols = ['Sex', 'Embarked', 'Title']
        string_indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep") for c in categorical_cols]
        one_hot_encoders = [OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_vec") for c in categorical_cols]

        # 2. Assemble all pre-engineered and newly encoded features into a single vector
        # Note: We now use the feature-engineered columns directly
        feature_cols = ['Pclass', 'Age_imputed', 'SibSp', 'Parch', 'Fare', 'IsAlone', 'FamilySize'] + [f"{c}_vec" for c in categorical_cols]
        vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
        
        # The pipeline stages now only contain the final modeling steps
        modeling_stages = [
            *string_indexers,
            *one_hot_encoders,
            vector_assembler
        ]
        
        print("   - Pipeline stages defined for final encoding and vector assembly.")

        # --- Define Models and Hyperparameter Grids (No Change) ---
        dt = DecisionTreeClassifier(labelCol="Survived", featuresCol="features")
        rf = RandomForestClassifier(labelCol="Survived", featuresCol="features")
        lr = LogisticRegression(labelCol="Survived", featuresCol="features")
        gbt = GBTClassifier(labelCol="Survived", featuresCol="features")

        dt_param_grid = ParamGridBuilder().addGrid(dt.maxDepth, [5, 7, 10]).build()
        rf_param_grid = ParamGridBuilder().addGrid(rf.numTrees, [20, 50]).addGrid(rf.maxDepth, [5, 7, 10]).build()
        lr_param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).build()
        gbt_param_grid = ParamGridBuilder().addGrid(gbt.maxDepth, [3, 5]).addGrid(gbt.maxIter, [10, 20]).build()
            
        models_to_tune = [
            ("GBTClassifier", gbt, gbt_param_grid),
            ("RandomForest", rf, rf_param_grid),
            ("DecisionTree", dt, dt_param_grid),
            ("LogisticRegression", lr, lr_param_grid)
        ]
        
        best_overall_model = None
        best_f1_score = -1.0
        best_param_map = None

        print("\n--- Starting Hyperparameter Tuning ---")
        for model_name, classifier, param_grid in models_to_tune:
            with mlflow.start_run(run_name=f"Tuning_{model_name}", nested=True):
                full_pipeline = Pipeline(stages=modeling_stages + [classifier])
                evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")
                cv = CrossValidator(estimator=full_pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
                cv_model = cv.fit(training_data)
                best_cv_f1_score = max(cv_model.avgMetrics)
                if best_cv_f1_score > best_f1_score:
                    best_f1_score = best_cv_f1_score
                    best_overall_model = cv_model.bestModel
                    best_param_map_raw = cv_model.getEstimatorParamMaps()[cv_model.avgMetrics.index(best_cv_f1_score)]
                    best_param_map = {p.name.split('_')[-1]: v for p, v in best_param_map_raw.items()}
                    print(f"   - ✨ New best model: {model_name} (F1: {best_f1_score:.4f})")
        
        print("\n--- Evaluating Best Overall Model ---")
        predictions = best_overall_model.transform(test_data)
        accuracy = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
        f1_final = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1").evaluate(predictions)
        precision = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
        recall = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
        auc = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC").evaluate(predictions)
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1_final, "precision": precision, "recall": recall, "auc": auc})
        print(f"   - Final Metrics: Accuracy={accuracy:.4f}, F1={f1_final:.4f}")

        # --- Artifact Logging and Model Registry (No Change) ---
        with open("metrics.json", "w") as f: json.dump({"accuracy": accuracy, "f1_score": f1_final}, f)
        mlflow.log_artifact("metrics.json")
        log_confusion_matrix(predictions)
        log_feature_importance(best_overall_model, predictions)
        with open("best_params.json", "w") as f: json.dump(best_param_map, f, indent=4)
        mlflow.log_artifact("best_params.json")
        best_overall_model.write().overwrite().save(model_output_path)
        model_name_registered = "TitanicSurvivalPredictor-Advanced"
        mlflow.spark.log_model(spark_model=best_overall_model, artifact_path="spark-model", registered_model_name=model_name_registered)
        client = MlflowClient()
        latest_version = client.get_latest_versions(model_name_registered, stages=["None"])[0].version
        client.transition_model_version_stage(name=model_name_registered, version=latest_version, stage="Staging", archive_existing_versions=True)
        print(f"   - ✅ Model registered as version {latest_version} and moved to 'Staging'.")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training on pre-engineered data")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--profile", required=True, help="The experiment profile from params.yaml to use.")
    args = parser.parse_args()
    main(args.train_data, args.model_out, args.profile)

