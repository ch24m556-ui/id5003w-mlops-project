# retraining_trigger.py
import pandas as pd
import os
import subprocess

def main():
    """
    Simulates the arrival of new data and triggers a DVC pipeline run.
    """
    print(" Starting automated retraining simulation...")

    # --- 1. Simulate new data arrival ---
    print("   - Simulating new data by taking a sample of 100 records from the test set.")
    # In a real scenario, this would be new data from a database, API, or data stream.
    try:
        new_data_sample = pd.read_csv("data/raw/test.csv").sample(n=100)
    except FileNotFoundError:
        print("   -  ERROR: Raw data not found. Please run 'dvc pull' first.")
        return

    # --- 2. (Optional) Induce Data Drift for Demonstration ---
    # To demonstrate that your data validation stage works, you can uncomment
    # the following lines. This will make the 'Age' of new passengers much higher,
    # causing the pipeline to stop at the 'validate_data' stage.
    
    # print("   - Artificially inducing drift in 'Age' for demonstration.")
    # new_data_sample['Age'] = new_data_sample['Age'] + 20 # Make everyone 20 years older

    # --- 3. Append new data to the existing training data ---
    # This action modifies a dependency of our DVC pipeline.
    print("   - Appending new data to 'data/raw/train.csv'.")
    new_data_sample.to_csv("data/raw/train.csv", mode='a', header=False, index=False)
    
    # --- 4. Trigger the DVC pipeline ---
    print("\n   - New data added. Triggering 'dvc repro' to retrain the model...")
    try:
        # We use subprocess to run the 'dvc repro' command from within Python.
        # `check=True` will raise an exception if the command fails.
        result = subprocess.run(
            ["dvc", "repro"], 
            check=True, 
            text=True, 
            capture_output=True
        )
        print("\n   -  DVC pipeline executed successfully!")
        print("\n--- DVC Output ---")
        print(result.stdout)
        print("--------------------")

    except subprocess.CalledProcessError as e:
        print("\n   -  DVC pipeline failed!")
        # Check if the failure was due to our intentional drift detection
        if "DRIFT DETECTED" in e.stdout or "DRIFT DETECTED" in e.stderr:
             print("   - REASON: The data validation stage correctly detected data drift and stopped the pipeline.")
             print("   - This demonstrates that the pipeline is robust against changes in data distribution.")
        else:
             print("   - An unexpected error occurred. See details below:")
        
        print("\n--- DVC Output (stdout) ---")
        print(e.stdout)
        print("\n--- DVC Output (stderr) ---")
        print(e.stderr)
        print("-----------------------------")
    
    print("\n Retraining simulation complete.")

if __name__ == "__main__":
    main()