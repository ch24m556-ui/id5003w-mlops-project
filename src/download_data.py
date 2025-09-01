import requests
import os
import pandas as pd
from pathlib import Path

def download_titanic_data():
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for Titanic dataset
    train_url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    
    # Download data
    print("Downloading Titanic dataset...")
    df = pd.read_csv(train_url)
    
    # Split into train and test (for demonstration)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save to files
    train_df.to_csv(data_dir / "titanic_train.csv", index=False)
    test_df.to_csv(data_dir / "titanic_test.csv", index=False)
    
    print("Data downloaded and saved successfully!")

if __name__ == "__main__":
    download_titanic_data()