# src/preprocess.py

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, trim

def main(train_input, test_input, output_path):
    """
    Main function to preprocess the Titanic dataset using Spark.

    :param train_input: Path to the raw training data CSV file.
    :param test_input: Path to the raw test data CSV file.
    :param output_path: Path to save the processed output data.
    """
    # 1. Initialize Spark Session
    spark = SparkSession.builder.appName("TitanicPreprocessing").getOrCreate()
    print("✅ Spark Session initialized.")

    # 2. Load the raw data
    train_df = spark.read.csv(train_input, header=True, inferSchema=True)
    test_df = spark.read.csv(test_input, header=True, inferSchema=True)
    print("✅ Raw training and test data loaded.")

    # --- 3. Feature Engineering & Preprocessing ---

    # Impute missing 'Age' with the mean age of the training data
    mean_age = train_df.select(avg("Age")).first()[0]
    train_df = train_df.fillna(mean_age, subset=["Age"])
    test_df = test_df.fillna(mean_age, subset=["Age"])
    print(f"   - Missing 'Age' imputed with mean value: {mean_age:.2f}")

    # Impute missing 'Embarked' with the most frequent value ('S')
    train_df = train_df.fillna("S", subset=["Embarked"])
    # (Test data has no missing 'Embarked', but we do it for consistency)
    test_df = test_df.fillna("S", subset=["Embarked"])
    print("   - Missing 'Embarked' imputed with mode value: 'S'")

    # Impute one missing 'Fare' in the test set with the mean fare
    mean_fare = train_df.select(avg("Fare")).first()[0]
    test_df = test_df.fillna(mean_fare, subset=["Fare"])
    print(f"   - Missing 'Fare' in test set imputed with mean value: {mean_fare:.2f}")

    # Drop columns that are not useful for the model
    columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    train_df = train_df.drop(*columns_to_drop)
    test_df = test_df.drop(*columns_to_drop)
    print(f"   - Dropped unused columns: {columns_to_drop}")

    # --- 4. Save the processed data ---

    # We will save the processed files as Parquet, which is an efficient
    # format for Spark and big data workflows.
    print("✅ Preprocessing complete. Saving processed data...")
    train_df.write.mode("overwrite").parquet(f"{output_path}/train")
    test_df.write.mode("overwrite").parquet(f"{output_path}/test")

    print(f"   - Processed training data saved to: {output_path}/train")
    print(f"   - Processed test data saved to: {output_path}/test")

    # Stop the Spark session
    spark.stop()


if __name__ == "__main__":
    # Set up argument parser to make the script reusable
    parser = argparse.ArgumentParser(description="Preprocess Titanic Data with Spark")
    parser.add_argument("--train_input", required=True, help="Path to raw train.csv")
    parser.add_argument("--test_input", required=True, help="Path to raw test.csv")
    parser.add_argument("--output_path", required=True, help="Path to save processed data")
    args = parser.parse_args()

    main(args.train_input, args.test_input, args.output_path)