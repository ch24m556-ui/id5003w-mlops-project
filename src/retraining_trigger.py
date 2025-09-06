# src/retraining_trigger.py
import pandas as pd
import os
import subprocess
import sys

def main():
    """
    Simulates the arrival of new data and triggers the DVC pipeline,
    then checks the output to confirm the conditional logic worked.
    """
    print("--- Starting Automated Retraining Simulation ---")

    # --- 1. Simulate new data arrival ---
    print("   - Simulating new data by sampling 100 records from the test set.")
    try:
        new_data_sample = pd.read_csv("data/raw/test.csv").sample(n=100, random_state=42)
    except FileNotFoundError:
        print("   -  ERROR: Raw data not found. Please run 'dvc pull' first.")
        sys.exit(1)

    # --- 2. Induce Data Drift (for testing the 'retrain' path) ---
    # To test if the pipeline correctly detects drift and retrains,
    # uncomment the following two lines. This makes the new data statistically
    # different from the baseline, forcing a retraining run.
    print("   -  Artificially inducing drift in 'Age' for demonstration.")
    new_data_sample['Age'] = new_data_sample['Age'] + 30

    # --- OPTION A: Induce Numerical Drift in 'Fare' (Recommended Test) ---
    # This simulates a scenario where ticket prices have drastically increased.
    # The preprocessing pipeline will not "fix" this, so the KS-test
    # in the validation stage is guaranteed to detect it.
    print("   - Inducing NUMERICAL drift in the 'Fare' column for demonstration.")
    new_data_sample['Fare'] = new_data_sample['Fare'] * np.random.uniform(3, 5, size=len(new_data_sample))


    # --- OPTION B: Induce Categorical Drift in 'Pclass' ---
    # This simulates a scenario where you suddenly only get first-class passengers.
    # The Chi-Squared test in your validation script will detect this.
    print("   - Inducing CATEGORICAL drift in the 'Pclass' column.")
    new_data_sample['Pclass'] = 1




    # --- 3. Append new data to the existing training data ---
    print("   - Appending new data to 'data/raw/train.csv'. This modifies a DVC dependency.")
    new_data_sample.to_csv("data/raw/train.csv", mode='a', header=False, index=False)
    
    # --- 4. Trigger the DVC pipeline ---
    print("\n   - New data added. Triggering 'dvc repro' to run the full pipeline...")
    try:
        result = subprocess.run(
            ["dvc", "repro"], 
            check=True, 
            text=True, 
            capture_output=True
        )
        
        print("\n---  DVC Pipeline Executed Successfully ---")
        
        # --- 5. Check the output to see what the pipeline did ---
        if "Skipping model retraining" in result.stdout:
            print("   - RESULT: The pipeline correctly detected NO data drift and SKIPPED retraining.")
            print("   - This is the efficient path for stable data.")
        elif "Proceeding with model retraining" in result.stdout:
            print("   - RESULT: The pipeline correctly detected DATA DRIFT and TRIGGERED retraining.")
            print("   - This is the robust path for changing data.")
        else:
            print("   - RESULT: Pipeline ran, but the specific outcome message was not found. Please check logs.")

        print("\n--- DVC Output ---")
        print(result.stdout)
        print("--------------------")

    except subprocess.CalledProcessError as e:
        print("\n---  DVC Pipeline Failed ---")
        print("   - An unexpected error occurred. See details below:")
        print("\n--- DVC Output (stdout) ---")
        print(e.stdout)
        print("\n--- DVC Output (stderr) ---")
        print(e.stderr)
        print("-----------------------------")
    
    print("\n--- Retraining Simulation Complete ---")

if __name__ == "__main__":
    main()

