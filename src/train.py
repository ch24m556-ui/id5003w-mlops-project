# src/train.py

# --- FIX: Environment Configuration for PySpark ---
# This block MUST be at the top of the file, before any Spark imports.
import os
import sys

# Set the PYSPARK_PYTHON environment variable to the path of the current Python executable.
# This ensures that Spark workers use the same Python interpreter as the driver.
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# --- End of FIX ---

import argparse
import mlflow
import json
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
    DecisionTreeClassificationModel,
    RandomForestClassificationModel
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from mlflow.tracking import MlflowClient

def log_feature_importance(spark_model, features, artifact_path="feature_importance.png"):
    """Extracts and logs a feature importance plot for tree-based models."""
    if not isinstance(spark_model.stages[-1], (DecisionTreeClassificationModel, RandomForestClassificationModel)):
        print("   - Skipping feature importance: model is not a tree-based type.")
        return

    model = spark_model.stages[-1]
    importances = model.featureImportances.toArray()
    
    feature_importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(artifact_path)
    plt.close()
    mlflow.log_artifact(artifact_path)
    print(f"   - ✅ Logged feature importance plot to {artifact_path}")


def log_confusion_matrix(predictions, artifact_path="confusion_matrix.png"):
    """Generates and logs a confusion matrix plot."""
    preds_and_labels = predictions.select(['prediction', 'Survived']).withColumn('label', col('Survived').cast('float')).select(['prediction', 'label']).collect()
    metrics = MulticlassMetrics(spark.sparkContext.parallelize(preds_and_labels))
    confusion_matrix = metrics.confusionMatrix().toArray()

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.savefig(artifact_path)
    plt.close()
    mlflow.log_artifact(artifact_path)
    print(f"   - ✅ Logged confusion matrix plot to {artifact_path}")


def main(train_data_path, model_output_path, spark_configs):
    """
    Main function to train models with parameterized Spark configurations.
    """
    global spark 
    spark_builder = SparkSession.builder.appName("SparkMLlibTuning")
    for key, value in spark_configs.items():
        spark_builder.config(key, value)
    spark = spark_builder.getOrCreate()
    print(f"✅ Spark Session initialized with custom configuration: {spark_configs}")

    mlflow.set_tracking_uri("./mlruns")

    with mlflow.start_run(run_name="Model Comparison with Spark Tuning") as parent_run:
        print("   - Logging Spark configuration to MLflow.")
        for key, value in spark_configs.items():
            mlflow.log_param(key, value)

        print("🚀 MLflow Parent Run Started for Model Comparison and Tuning.")
        df = spark.read.parquet(train_data_path)
        (training_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)
        print("   - Data loaded and split for training and validation.")

        categorical_cols = ['Sex', 'Embarked']
        numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        feature_stages = []
        for col_name in categorical_cols:
            indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index", handleInvalid="skip")
            encoder = OneHotEncoder(inputCol=f"{col_name}_index", outputCol=f"{col_name}_vec")
            feature_stages += [indexer, encoder]
        
        feature_cols = numerical_cols + [f"{c}_vec" for c in categorical_cols]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
        feature_stages.append(assembler)
        print("   - Feature engineering pipeline defined.")

        dt = DecisionTreeClassifier(labelCol="Survived", featuresCol="features")
        rf = RandomForestClassifier(labelCol="Survived", featuresCol="features")
        lr = LogisticRegression(labelCol="Survived", featuresCol="features")

        dt_param_grid = ParamGridBuilder().addGrid(dt.maxDepth, [3, 5, 7]).addGrid(dt.maxBins, [32, 40]).build()
        rf_param_grid = ParamGridBuilder().addGrid(rf.numTrees, [10, 20, 30]).addGrid(rf.maxDepth, [3, 5, 7]).build()
        lr_param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1, 0.5]).addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]).build()
            
        models_to_tune = [
            ("DecisionTree", dt, dt_param_grid),
            ("RandomForest", rf, rf_param_grid),
            ("LogisticRegression", lr, lr_param_grid),
        ]
        
        best_overall_model = None
        best_f1_score = -1.0
        best_param_map = None

        print("\n--- Starting Hyperparameter Tuning for Each Model ---")
        for model_name, classifier, param_grid in models_to_tune:
            with mlflow.start_run(run_name=f"Tuning_{model_name}", nested=True) as child_run:
                print(f"\n   Tuning {model_name}...")
                pipeline = Pipeline(stages=feature_stages + [classifier])
                evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")
                cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3, parallelism=2)
                cv_model = cv.fit(training_data)
                
                best_cv_f1_score = max(cv_model.avgMetrics)
                print(f"   - Best cross-validated F1 score for {model_name}: {best_cv_f1_score:.4f}")
                mlflow.log_metric("best_cv_f1_score", best_cv_f1_score)
                
                if best_cv_f1_score > best_f1_score:
                    best_f1_score = best_cv_f1_score
                    best_overall_model = cv_model.bestModel
                    best_param_map_raw = cv_model.getEstimatorParamMaps()[cv_model.avgMetrics.index(best_cv_f1_score)]
                    best_param_map = {p.name: v for p, v in best_param_map_raw.items()}
                    print(f"   - ✨ New best overall model found: {model_name}")

        print("\n--- Tuning Complete. Evaluating Best Overall Model ---")
        predictions = best_overall_model.transform(test_data)

        # --- Calculate and Log All Metrics ---
        accuracy = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
        f1_final = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1").evaluate(predictions)
        precision = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
        recall = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
        auc = BinaryClassificationEvaluator(labelCol="Survived", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(predictions)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1_final)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("auc", auc)
        print(f"   - Final Metrics on Test Set: Accuracy={accuracy:.4f}, F1={f1_final:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, AUC={auc:.4f}")

        metrics = {"accuracy": accuracy, "f1_score": f1_final, "precision": precision, "recall": recall, "auc": auc}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact("metrics.json")
        
        # --- Log Artifacts ---
        log_confusion_matrix(predictions)
        log_feature_importance(best_overall_model, feature_cols)
        
        with open("best_params.json", "w") as f:
            json.dump(best_param_map, f, indent=4)
        mlflow.log_artifact("best_params.json")
        print("   - ✅ Logged best parameters to best_params.json")

        best_overall_model.write().overwrite().save(model_output_path)
        print(f"   - ✅ Best Spark model saved to {model_output_path}")

        model_name_registered = "TitanicSparkModel-MultiTuned"
        mlflow.spark.log_model(
            spark_model=best_overall_model,
            artifact_path="spark-model",
            registered_model_name=model_name_registered
        )
        
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
    parser = argparse.ArgumentParser(description="Tune and train models with different Spark configs")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--spark_driver_memory", required=True)
    parser.add_argument("--spark_executor_cores", required=True, type=int)
    parser.add_argument("--spark_shuffle_partitions", required=True, type=int)
    
    args = parser.parse_args()

    spark_configurations = {
        "spark.driver.memory": args.spark_driver_memory,
        "spark.executor.cores": args.spark_executor_cores,
        "spark.sql.shuffle.partitions": args.spark_shuffle_partitions
    }

    main(args.train_data, args.model_out, spark_configurations)

