# src/trigger_training.py
import json
import subprocess
import sys
import os
import shutil

def main():
    """
    Reads the drift status and model existence, and decides whether to run 
    the main training script or to skip it.
    """
    # --- 1. Get the declared model output path from arguments ---
    try:
        model_out_index = sys.argv.index("--model_out") + 1
        model_out_path = sys.argv[model_out_index]
    except (ValueError, IndexError):
        print("❌ ERROR: Could not find --model_out argument. This is required.")
        sys.exit(1)

    # --- 2. Read the drift status ---
    try:
        with open("drift_status.json", "r") as f:
            status = json.load(f)
        drift_detected = status.get("drift_detected", False)
    except FileNotFoundError:
        print("❌ ERROR: drift_status.json not found. Did the validation stage run?")
        sys.exit(1)

    # --- 3. Check if the model artifact already exists and is not empty ---
    model_exists = os.path.exists(model_out_path) and len(os.listdir(model_out_path)) > 0
    
    # --- 4. Decide whether to trigger training ---
    # Trigger training if drift is detected OR if the model doesn't exist yet.
    if drift_detected or not model_exists:
        if drift_detected:
            print("✅ TRIGGER: Data drift detected. Proceeding with model retraining.")
        if not model_exists:
            print("✅ TRIGGER: Model artifact is missing or empty. Proceeding with initial training.")

        # Construct the command to call train.py, passing along all original arguments
        command = ["python", "src/train.py"] + sys.argv[1:] 

        try:
            subprocess.run(command, check=True)
            print("   - (OK) Model training script executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"   - ❌ ERROR: Model training script failed with exit code {e.returncode}.")
            sys.exit(1)

    else:
        print("✅ SKIP: No data drift detected and a valid model already exists.")
        print("   - Skipping model retraining to save resources.")
        
        # DVC requires outputs exist, so we ensure they are not removed.
        # No need to create placeholders if the model already exists.
        print("   - Existing artifacts are preserved.")

if __name__ == "__main__":
    main()