# src/validate_data.py
import argparse
import json
import sys
import pandas as pd
from pyspark.sql import SparkSession
from scipy.stats import ks_2samp, chi2_contingency

def main(new_data_path, baseline_sample_path, output_status_path):
    """
    Performs comprehensive drift detection on both numerical and categorical features.
    """
    spark = SparkSession.builder.appName("ComprehensiveDriftDetector").getOrCreate()
    print("(OK) Spark Session initialized.")

    new_df = spark.read.parquet(new_data_path)
    baseline_df = spark.read.parquet(baseline_sample_path)
    print("(OK) New data and baseline sample loaded.")

    drift_detected = False
    
    # Define features to monitor for drift
    numerical_features = ["Age_imputed", "Fare", "FamilySize", "SibSp", "Parch"]
    categorical_features = ["Pclass", "Sex", "Embarked", "Title", "IsAlone"]

    # --- 1. Numerical Drift Detection (KS Test) ---
    print("\n--- Checking for drift in numerical features ---")
    for feature in numerical_features:
        if feature not in new_df.columns or feature not in baseline_df.columns:
            print(f"   - WARNING: Feature '{feature}' not found, skipping.")
            continue
            
        print(f"   - Performing KS test for '{feature}'...")
        new_series = new_df.select(feature).toPandas()[feature].dropna()
        baseline_series = baseline_df.select(feature).toPandas()[feature].dropna()
        
        ks_statistic, p_value = ks_2samp(new_series, baseline_series)
        
        print(f"     - KS test for {feature}: p-value = {p_value:.4f}")
        if p_value < 0.05:
            print(f"     - (DRIFT DETECTED) The distribution of '{feature}' has changed significantly.")
            drift_detected = True
        else:
            print(f"     - (OK) No significant drift detected in '{feature}'.")

    # --- 2. Categorical Drift Detection (Chi-Squared Test) ---
    print("\n--- Checking for drift in categorical features ---")
    for feature in categorical_features:
        if feature not in new_df.columns or feature not in baseline_df.columns:
            print(f"   - WARNING: Feature '{feature}' not found, skipping.")
            continue

        print(f"   - Performing Chi-Squared test for '{feature}'...")
        new_counts = new_df.groupBy(feature).count().toPandas().set_index(feature)
        baseline_counts = baseline_df.groupBy(feature).count().toPandas().set_index(feature)
        
        # Create a contingency table
        contingency_table = pd.concat([new_counts, baseline_counts], axis=1).fillna(0)
        contingency_table.columns = ['new', 'baseline']

        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        print(f"     - Chi-Squared test for {feature}: p-value = {p_value:.4f}")
        if p_value < 0.05:
            print(f"     - (DRIFT DETECTED) The distribution of '{feature}' has changed significantly.")
            drift_detected = True
        else:
            print(f"     - (OK) No significant drift detected in '{feature}'.")

    # --- 3. Save the final drift status ---
    print(f"\n Drift detection complete. Overall drift detected: {drift_detected}")
    with open(output_status_path, "w") as f:
        json.dump({"drift_detected": drift_detected}, f, indent=4)
    print(f"   - Status saved to {output_status_path}")
    
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Data Drift Detection")
    parser.add_argument("--new_data", required=True, help="Path to new data (Parquet format)")
    parser.add_argument("--baseline_sample", required=True, help="Path to baseline data (Parquet format)")
    parser.add_argument("--output_status_path", required=True, help="Path to save the drift status JSON file")
    args = parser.parse_args()
    main(args.new_data, args.baseline_sample, args.output_status_path)

