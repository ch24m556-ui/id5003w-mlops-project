# src/create_baseline.py
import argparse
from pyspark.sql import SparkSession

def main(train_input, baseline_output_path):
    """
    Takes the raw training data, creates a representative sample,
    and saves it as the official baseline for drift detection.
    This is typically run only once, or when the baseline needs to be formally updated.
    """
    spark = SparkSession.builder.appName("CreateBaseline").getOrCreate()
    print("(OK) Spark Session initialized for baseline creation.")

    train_df = spark.read.csv(train_input, header=True, inferSchema=True)
    
    # Create the baseline sample from the raw, trusted training data
    baseline_sample = train_df.sample(withReplacement=False, fraction=0.2, seed=42)
    
    # Save the baseline sample
    baseline_sample.write.mode("overwrite").parquet(baseline_output_path)
    print(f"✅ Baseline sample created and saved to: {baseline_output_path}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a baseline data sample for drift detection")
    parser.add_argument("--train_input", required=True, help="Path to the raw, trusted train.csv")
    parser.add_argument("--baseline_output", required=True, help="Path to save the baseline sample artifact")
    args = parser.parse_args()
    main(args.train_input, args.baseline_output)
