# src/data_processing.py

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, expr, when, regexp_extract
from pyspark.sql.types import DoubleType

def main(train_input, test_input, output_path, baseline_sample_path):
    """
    Main function to preprocess and engineer features for the Titanic dataset using Spark.
    """
    spark = SparkSession.builder.appName("TitanicFeatureEngineering").getOrCreate()
    print("(OK) Spark Session initialized.")

    train_df = spark.read.csv(train_input, header=True, inferSchema=True)
    test_df = spark.read.csv(test_input, header=True, inferSchema=True)
    print("(OK) Raw training and test data loaded.")

    print("\n--- Starting Robust Data Cleaning ---")
    numeric_cols_to_cast = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Survived"]
    for col_name in numeric_cols_to_cast:
        if col_name in train_df.columns:
            train_df = train_df.withColumn(col_name, expr(f"try_cast({col_name} as double)"))
        if col_name in test_df.columns and col_name != "Survived":
             test_df = test_df.withColumn(col_name, expr(f"try_cast({col_name} as double)"))
    print("   - (FIX) Robustly cast all numeric columns to DoubleType, handling malformed strings.")
    
    combined_df = train_df.unionByName(test_df, allowMissingColumns=True)

    print("\n--- Starting Advanced Feature Engineering ---")
    combined_df = combined_df.withColumn("Title", regexp_extract(col("Name"), " ([A-Za-z]+)\\.", 1))
    
    # --- THIS IS THE FIX ---
    # Replace any empty strings created by the regex with a placeholder category 'Other'.
    # This prevents the StringIndexer from failing on invalid category names.
    combined_df = combined_df.withColumn("Title", when(col("Title") == "", "Other").otherwise(col("Title")))
    print("   - (FEAT) 'Title' extracted from 'Name' and cleaned.")
    # --- End of FIX ---

    title_ages = combined_df.groupBy("Title").agg(avg("Age").alias("mean_age"))
    combined_df = combined_df.join(title_ages, on="Title", how="left")
    combined_df = combined_df.withColumn("Age_imputed", when(col("Age").isNull(), col("mean_age")).otherwise(col("Age"))).drop("mean_age")
    print("   - (FEAT) Missing 'Age' values imputed based on title-specific averages.")
    combined_df = combined_df.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)
    combined_df = combined_df.withColumn("IsAlone", when(col("FamilySize") == 1, 1).otherwise(0))
    print("   - (FEAT) 'FamilySize' and 'IsAlone' features created.")

    print("\n--- Starting Standard Imputation ---")
    embarked_mode = combined_df.groupBy("Embarked").count().orderBy("count", ascending=False).first()[0]
    combined_df = combined_df.fillna({"Embarked": embarked_mode})
    print(f"   - Missing 'Embarked' imputed with mode value: '{embarked_mode}'")
    fare_mean = combined_df.select(avg("Fare")).first()[0]
    combined_df = combined_df.fillna({"Fare": fare_mean})
    print(f"   - Missing 'Fare' imputed with mean value: {fare_mean:.2f}")

    columns_to_drop = ["PassengerId", "Ticket", "Cabin", "Name", "Age"]
    combined_df = combined_df.drop(*columns_to_drop)
    print(f"\n   - Dropped unused columns: {columns_to_drop}")

    final_train_df = combined_df.filter(col("Survived").isNotNull())
    final_test_df = combined_df.filter(col("Survived").isNull())

    print("\n(OK) Preprocessing and feature engineering complete. Saving data...")
    final_train_df.write.mode("overwrite").parquet(f"{output_path}/train")
    final_test_df.write.mode("overwrite").parquet(f"{output_path}/test")
    print(f"   - Processed data saved to: {output_path}")

    baseline_sample = final_train_df.sample(withReplacement=False, fraction=0.2, seed=42)
    baseline_sample.write.mode("overwrite").parquet(baseline_sample_path)
    print(f"   - Baseline sample saved to: {baseline_sample_path}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and engineer features for Titanic Data with Spark")
    parser.add_argument("--train_input", required=True, help="Path to raw train.csv")
    parser.add_argument("--test_input", required=True, help="Path to raw test.csv")
    parser.add_argument("--output_path", required=True, help="Path to save processed data")
    parser.add_argument("--baseline_sample_path", required=True, help="Path to save baseline sample for drift detection")
    args = parser.parse_args()
    main(args.train_input, args.test_input, args.output_path, args.baseline_sample_path)

