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
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
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
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

# --- Helper Functions for Artifacts ---
def log_feature_importance(spark_model, predictions, artifact_path="feature_importance.png"):
    """Extracts and logs a feature importance plot for tree-based models."""
    if not isinstance(spark_model.stages[-1], (DecisionTreeClassificationModel, RandomForestClassificationModel)):
        print("   - Skipping feature importance: model is not a tree-based type.")
        return
    
    assembler_stage = next(s for s in spark_model.stages if isinstance(s, VectorAssembler))
    feature_attrs = predictions.schema[assembler_stage.getOutputCol()].metadata["ml_attr"]["attrs"]
    feature_names = []
    if "numeric" in feature_attrs:
        feature_names.extend([attr["name"] for attr in feature_attrs["numeric"]])
    if "binary" in feature_attrs:
        feature_names.extend([attr["name"] for attr in feature_attrs["binary"]])

    model = spark_model.stages[-1]
    importances = model.featureImportances.toArray()

    if len(feature_names) != len(importances):
        print(f"   -  Warning: Mismatch between feature names ({len(feature_names)}) and importances ({len(importances)}). Skipping plot.")
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
    print(f"   -  Logged feature importance plot to {artifact_path}")

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
    print(f"   -  Logged confusion matrix plot to {artifact_path}")

def log_roc_curve(predictions, artifact_path="roc_curve.png"):
    """Generates and logs a ROC curve plot."""
    # Extract probability and label columns
    extract_prob = udf(lambda v: float(v[1]), DoubleType())
    probs_and_labels = predictions.select(
        extract_prob('probability').alias('probability'),
        col('Survived').cast('float').alias('label')
    ).toPandas()

    fpr, tpr, _ = roc_curve(probs_and_labels['label'], probs_and_labels['probability'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(artifact_path)
    plt.close()
    mlflow.log_artifact(artifact_path)
    mlflow.log_metric("roc_auc", roc_auc)
    print(f"   -  Logged ROC curve plot to {artifact_path}")

def log_precision_recall_curve(predictions, artifact_path="precision_recall_curve.png"):
    """Generates and logs a precision-recall curve plot."""
    extract_prob = udf(lambda v: float(v[1]), DoubleType())
    probs_and_labels = predictions.select(
        extract_prob('probability').alias('probability'),
        col('Survived').cast('float').alias('label')
    ).toPandas()

    precision, recall, _ = precision_recall_curve(probs_and_labels['label'], probs_and_labels['probability'])
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.savefig(artifact_path)
    plt.close()
    mlflow.log_artifact(artifact_path)
    mlflow.log_metric("pr_auc", pr_auc)
    print(f"   -  Logged precision-recall curve plot to {artifact_path}")

def log_classification_report(predictions, artifact_path="classification_report.json"):
    """Generates and logs a comprehensive classification report."""
    preds_and_labels = predictions.select(['prediction', 'Survived']).withColumn('label', col('Survived').cast('float')).select(['prediction', 'label']).rdd.map(tuple)
    metrics = MulticlassMetrics(preds_and_labels)
    
    # Calculate various metrics
    report = {
        "accuracy": metrics.accuracy,
        "weightedPrecision": metrics.weightedPrecision,
        "weightedRecall": metrics.weightedRecall,
        "weightedFMeasure": metrics.weightedFMeasure(),
        "weightedFMeasure_beta": metrics.weightedFMeasure(2.0),
        "falsePositiveRate_0": metrics.falsePositiveRate(0.0),
        "falsePositiveRate_1": metrics.falsePositiveRate(1.0),
        "truePositiveRate_0": metrics.truePositiveRate(0.0),
        "truePositiveRate_1": metrics.truePositiveRate(1.0)
    }
    
    with open(artifact_path, "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact(artifact_path)
    
    # Log individual metrics
    for metric_name, value in report.items():
        mlflow.log_metric(metric_name, value)
    
    print(f"   -  Logged classification report to {artifact_path}")

# --- Main Training Logic ---
def main(train_data_path, model_output_path, profile_name):
    """
    Main function to train models on pre-processed, feature-engineered data.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    spark_configs = params["spark_configs"][profile_name]

    # Initialize Spark session
    spark_builder = SparkSession.builder.appName("SimplifiedSparkMLTraining")
    for key, value in spark_configs.items():
        spark_builder = spark_builder.config(key, value)
    spark = spark_builder.getOrCreate()

    print(f" Spark Session initialized with '{profile_name}' profile: {spark_configs}")

    mlflow.set_tracking_uri("./mlruns")
    
    # Set experiment tags
    tags = {
        "project": "Titanic Survival Prediction",
        "team": "Data Science",
        "profile": profile_name,
        "dataset": "Titanic",
        "framework": "PySpark"
    }
    
    with mlflow.start_run(run_name=f"Training with {profile_name} profile") as parent_run:
        # Log parameters and tags
        mlflow.log_params(spark_configs)
        mlflow.log_param("experiment_profile", profile_name)
        mlflow.log_param("train_data_path", train_data_path)
        mlflow.log_param("model_output_path", model_output_path)
        mlflow.set_tags(tags)
        
        print(f" MLflow Run Started for profile: {profile_name}.")

        df = spark.read.parquet(train_data_path)
        (training_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)
        print("   - Pre-processed data loaded and split for training and validation.")

        # Define the Simplified ML Pipeline
        print("\n--- Defining Simplified Modeling Pipeline ---")
        categorical_cols = ['Sex', 'Embarked', 'Title']
        string_indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep") for c in categorical_cols]
        one_hot_encoders = [OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_vec") for c in categorical_cols]
        feature_cols = ['Pclass', 'Age_imputed', 'SibSp', 'Parch', 'Fare', 'IsAlone', 'FamilySize'] + [f"{c}_vec" for c in categorical_cols]
        vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
        modeling_stages = [*string_indexers, *one_hot_encoders, vector_assembler]
        
        # Log feature information
        feature_info = {
            "categorical_features": categorical_cols,
            "numerical_features": ['Pclass', 'Age_imputed', 'SibSp', 'Parch', 'Fare', 'IsAlone', 'FamilySize'],
            "all_features": feature_cols
        }
        with open("feature_info.json", "w") as f:
            json.dump(feature_info, f, indent=4)
        mlflow.log_artifact("feature_info.json")
        
        print("   - Pipeline stages defined for final encoding and vector assembly.")

        # Define Models and Hyperparameter Grids
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
        
        best_overall_model, best_f1_score, best_param_map, best_model_name = None, -1.0, None, ""

        print("\n--- Starting Hyperparameter Tuning ---")
        for model_name, classifier, param_grid in models_to_tune:
            with mlflow.start_run(run_name=f"Tuning_{model_name}", nested=True):
                full_pipeline = Pipeline(stages=modeling_stages + [classifier])
                evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")
                cv = CrossValidator(estimator=full_pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
                cv_model = cv.fit(training_data)
                best_cv_f1_score = max(cv_model.avgMetrics)
                
                # Log metrics for this model
                mlflow.log_metric("best_cv_f1_score", best_cv_f1_score)
                mlflow.log_param("model_type", model_name)
                
                if best_cv_f1_score > best_f1_score:
                    best_f1_score = best_cv_f1_score
                    best_overall_model = cv_model.bestModel
                    best_param_map_raw = cv_model.getEstimatorParamMaps()[cv_model.avgMetrics.index(best_cv_f1_score)]
                    best_param_map = {p.name.split('_')[-1]: v for p, v in best_param_map_raw.items()}
                    best_model_name = model_name
                    print(f"   - ✨ New best model: {model_name} (CV F1: {best_f1_score:.4f})")
        
        # Log best model information
        mlflow.log_param("best_model_type", best_model_name)
        mlflow.log_params(best_param_map)
        
        print("\n--- Evaluating Best Overall Model ---")
        predictions = best_overall_model.transform(test_data)
        
        # Calculate multiple metrics
        accuracy = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
        f1_final = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1").evaluate(predictions)
        precision = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
        recall = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
        auc_roc = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC").evaluate(predictions)
        auc_pr = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderPR").evaluate(predictions)
        
        # Log all metrics
        metrics = {
            "accuracy": accuracy,
            "f1_score": f1_final,
            "precision": precision,
            "recall": recall,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr
        }
        mlflow.log_metrics(metrics)
        print(f"   - Final Metrics: Accuracy={accuracy:.4f}, F1={f1_final:.4f}, AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}")

        # Artifact Logging and Model Registry
        with open("metrics.json", "w") as f: 
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact("metrics.json")
        
        # Log various artifacts
        log_confusion_matrix(predictions)
        log_feature_importance(best_overall_model, predictions)
        log_roc_curve(predictions)
        log_precision_recall_curve(predictions)
        log_classification_report(predictions)
        
        with open("best_params.json", "w") as f: 
            json.dump(best_param_map, f, indent=4)
        mlflow.log_artifact("best_params.json")
        
        # Save and register model with additional information
        best_overall_model.write().overwrite().save(model_output_path)
        
        # Create model signature
        input_example = training_data.limit(5).toPandas()
        signature = infer_signature(input_example, best_overall_model.transform(training_data.limit(5)).toPandas())
        
        # Log model with additional metadata
        model_name_registered = "TitanicSurvivalPredictor-Advanced"
        mlflow.spark.log_model(
            spark_model=best_overall_model,
            artifact_path="spark-model",
            registered_model_name=model_name_registered,
            signature=signature,
            input_example=input_example,
            conda_env=_mlflow_conda_env(
                additional_conda_deps=None,
                additional_pip_deps=["pyspark=={}".format(spark.version)],
                additional_conda_channels=None
            )
        )
        
        # Transition model to staging
        client = MlflowClient()
        latest_version = client.get_latest_versions(model_name_registered, stages=["None"])[0].version
        client.transition_model_version_stage(
            name=model_name_registered, 
            version=latest_version, 
            stage="Staging", 
            archive_existing_versions=True
        )
        
        # Add model description
        model_desc = f"""
        Best performing model: {best_model_name}
        Achieved {f1_final:.4f} F1 score on test data.
        Trained with parameters: {best_param_map}
        """
        client.update_model_version(
            name=model_name_registered,
            version=latest_version,
            description=model_desc
        )
        
        print(f"   -  Model registered as version {latest_version} and moved to 'Staging'.")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training on pre-engineered data")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--profile", required=True, help="The experiment profile from params.yaml to use.")
    args = parser.parse_args()
    main(args.train_data, args.model_out, args.profile)