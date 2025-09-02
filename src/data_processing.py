# src/data_processing.py

import argparse
from pyspark.sql import SparkSession
# Import 'expr' to use advanced Spark SQL functions like try_cast
from pyspark.sql.functions import col, avg, expr
import json
from pyspark.sql.types import DoubleType, IntegerType, FloatType, LongType

def main(train_input, test_input, output_path, baseline_sample_path):
    """
    Main function to preprocess the Titanic dataset using Spark.
    """
    spark = SparkSession.builder.appName("TitanicPreprocessing").getOrCreate()
    print("(OK) Spark Session initialized.")

    train_df = spark.read.csv(train_input, header=True, inferSchema=True)
    test_df = spark.read.csv(test_input, header=True, inferSchema=True)
    print("(OK) Raw training and test data loaded.")

    # --- THIS IS THE DEFINITIVE FIX ---
    # We will explicitly cast ALL columns that should be numeric to DoubleType.
    # This enforces a consistent schema and prevents data type errors downstream.
    # The try_cast function robustly handles any malformed non-numeric values.
    numeric_cols_to_cast = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

    for col_name in numeric_cols_to_cast:
        train_df = train_df.withColumn(col_name, expr(f"try_cast({col_name} as double)"))
        test_df = test_df.withColumn(col_name, expr(f"try_cast({col_name} as double)"))

    print("   - (FIX) Robustly cast all numeric columns to DoubleType.")

    # --- Impute missing values ---
    mean_age = train_df.select(avg("Age")).first()[0]
    train_df = train_df.fillna(mean_age, subset=["Age"])
    test_df = test_df.fillna(mean_age, subset=["Age"])
    print(f"   - Missing 'Age' imputed with mean value: {mean_age:.2f}")

    train_df = train_df.fillna("S", subset=["Embarked"])
    test_df = test_df.fillna("S", subset=["Embarked"])
    print("   - Missing 'Embarked' imputed with mode value: 'S'")

    mean_fare = train_df.select(avg("Fare")).first()[0]
    test_df = test_df.fillna(mean_fare, subset=["Fare"])
    print(f"   - Missing 'Fare' in test set imputed with mean value: {mean_fare:.2f}")

    # --- Drop unused columns ---
    columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    train_df = train_df.drop(*columns_to_drop)
    test_df = test_df.drop(*columns_to_drop)
    print(f"   - Dropped unused columns: {columns_to_drop}")

    # --- Save the main processed data ---
    print("(OK) Preprocessing complete. Saving processed data...")
    train_df.write.mode("overwrite").parquet(f"{output_path}/train")
    test_df.write.mode("overwrite").parquet(f"{output_path}/test")
    print(f"   - Processed data saved to: {output_path}")

    # --- Create and save a baseline data sample ---
    print("   - Creating and saving a baseline data sample for drift detection...")
    baseline_sample = train_df.sample(withReplacement=False, fraction=0.2, seed=42)
    baseline_sample.write.mode("overwrite").parquet(baseline_sample_path)
    print(f"   - Baseline sample saved to: {baseline_sample_path}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Titanic Data with Spark")
    parser.add_argument("--train_input", required=True, help="Path to raw train.csv")
    parser.add_argument("--test_input", required=True, help="Path to raw test.csv")
    parser.add_argument("--output_path", required=True, help="Path to save processed data")
    parser.add_argument("--baseline_sample_path", required=True, help="Path to save the baseline data sample")
    args = parser.parse_args()
    main(args.train_input, args.test_input, args.output_path, args.baseline_sample_path)

