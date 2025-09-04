# src/trigger_training.py
import json
import subprocess
import sys
import os
import shutil

def main():
    """
    Reads the drift status and decides whether to run the main training script
    or to skip it and create placeholder artifacts.
    """
    # --- Read the drift status ---
    try:
        with open("drift_status.json", "r") as f:
            status = json.load(f)
    except FileNotFoundError:
        print("ERROR: drift_status.json not found. Did the validation stage run?")
        sys.exit(1)

    if status.get("drift_detected", False):
        print("(OK) Data drift detected. Proceeding with model retraining.")
        
        # --- Trigger the main training script ---
        # This script re-uses the command-line arguments it received from DVC
        # but calls train.py instead of itself.
        
        # Construct the command to call train.py, passing along all original arguments
        command = [
            "python", "src/train.py"
        ] + sys.argv[1:] 

        try:
            subprocess.run(command, check=True)
            print("(OK) Model training script executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Model training script failed with exit code {e.returncode}.")
            sys.exit(1)

    else:
        print("(OK) No data drift detected. Skipping model retraining to save resources.")
        
        # --- Create placeholder artifacts so DVC doesn't complain ---
        # DVC requires that all declared outputs exist after a stage runs.
        print("   - Creating placeholder artifacts for DVC.")
        
        # Find the model output path from the arguments
        # Arguments are passed like: --model_out, path/to/model
        try:
            model_out_index = sys.argv.index("--model_out") + 1
            model_out_path = sys.argv[model_out_index]
            
            # Create an empty directory for the model artifact
            if os.path.exists(model_out_path):
                shutil.rmtree(model_out_path)
            os.makedirs(model_out_path, exist_ok=True)

            # Create a dummy metrics file
            with open("metrics.json", "w") as f:
                json.dump({"status": "retraining_skipped", "reason": "no_drift"}, f)

        except (ValueError, IndexError):
            print("   - WARNING: Could not find --model_out argument. Cannot create placeholder artifacts.")

if __name__ == "__main__":
    main()

