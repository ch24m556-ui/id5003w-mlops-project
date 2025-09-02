# src/validate_data.py
import argparse
import sys
from pyspark.sql import SparkSession
from scipy.stats import ks_2samp

def main(new_data_path, baseline_sample_path):
    """
    Compares the distribution of new data against a baseline sample
    to detect data drift.
    """
    spark = SparkSession.builder.appName("DataDriftDetector").getOrCreate()
    
    # Load the new data and the baseline reference sample
    new_df = spark.read.parquet(new_data_path)
    baseline_df = spark.read.parquet(baseline_sample_path)
    
    print("(OK) New data and baseline sample loaded.")
    
    drift_detected = False
    
    # --- Perform Kolmogorov-Smirnov (KS) test for 'Age' feature ---
    print("   - Performing KS test for 'Age' feature...")
    new_age_series = new_df.select('Age').toPandas()['Age']
    baseline_age_series = baseline_df.select('Age').toPandas()['Age']
    
    # This test now compares two real data distributions
    ks_statistic, p_value = ks_2samp(new_age_series, baseline_age_series)
    
    print(f"   - KS test for Age: p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("   - (DRIFT DETECTED) The distribution of 'Age' has changed significantly.")
        drift_detected = True
    else:
        print("   - (OK) No significant drift detected in 'Age' feature.")

    # --- You can add more tests for other features here ---

    if drift_detected:
        print("\nPipeline stopped due to detected data drift.")
        sys.exit(1) # Stop the DVC pipeline
    else:
        print("\nData validation successful. Proceeding with training.")
        sys.exit(0) # Allow the pipeline to continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Drift Detection")
    parser.add_argument("--new_data", required=True, help="Path to new processed data")
    parser.add_argument("--baseline_sample", required=True, help="Path to the baseline data sample")
    args = parser.parse_args()
    main(args.new_data, args.baseline_sample)

