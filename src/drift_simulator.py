# src/simulate_drift.py
import pandas as pd
import numpy as np
import sys

def main():
    """
    Simulates the arrival of new data with intentionally induced drift
    to test the MLOps pipeline's data validation stage.
    """
    print("--- Starting Data Drift Simulation ---")

    try:
        # In a real scenario, this would be a new batch of data.
        # We simulate this by sampling from the existing test set.
        new_data_sample = pd.read_csv("data/raw/test.csv").sample(n=150, random_state=42)
        print("   - (1/3) Simulated new data arrival by sampling 150 records.")
    except FileNotFoundError:
        print("   - ❌ ERROR: Raw data not found. Please run 'dvc pull' first.")
        sys.exit(1)

    # --- CHOOSE ONE TYPE OF DRIFT TO SIMULATE ---
    # To test your pipeline, uncomment ONE of the blocks below.

    # --- Option A: Induce Numerical Drift in 'Age' ---
    # This makes all new passengers significantly older, which the
    # KS-test in your validation script will easily detect.
    print("   - (2/3) Inducing NUMERICAL drift in the 'Age' column.")
    new_data_sample['Age'] = new_data_sample['Age'] + np.random.uniform(25, 40, size=len(new_data_sample))


    # --- Induce Categorical Drift in 'Pclass' ---
    # This simulates a scenario where you suddenly only get first-class passengers.
    # The Chi-Squared test in your validation script will detect this change in distribution.
    print("   - (2/3) Inducing CATEGORICAL drift in the 'Pclass' column.")
    new_data_sample['Pclass'] = 1


    # --- 3. Append drifted data to the training file ---
    # This action modifies a dependency of the DVC pipeline.
    new_data_sample.to_csv("data/raw/train.csv", mode='a', header=False, index=False)
    print("   - (3/3) Appended drifted data to 'data/raw/train.csv'.")
    
    print("\n--- ✅ Simulation Complete ---")
    print("   - The raw training data has been modified.")
    print("   - Now, run 'dvc repro' to see the pipeline detect the drift.")

if __name__ == "__main__":
    main()