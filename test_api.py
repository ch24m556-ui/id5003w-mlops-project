# test_api.py
import requests
import json
import os

# Configuration
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

def test_health_endpoint():
    """Test the health endpoint."""
    try:
        response = requests.get(f"{API_URL}/health")
        assert response.status_code == 200, f"Health check failed with status {response.status_code}"
        
        data = response.json()
        assert data["status"] == "healthy", "Health status not healthy"
        assert data["spark_running"] == True, "Spark not running"
        assert data["model_loaded"] == True, "Model not loaded"
        
        print(" Health endpoint test passed")
        return True
    except Exception as e:
        print(f" Health endpoint test failed: {e}")
        return False

def test_predict_endpoint_survives():
    """Test prediction for a passenger who should survive."""
    payload = {
        "Pclass": 1,
        "Age": 38.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 71.2833,
        "Sex": "female",
        "Embarked": "C",
        "Name": "Cumings, Mrs. John Bradley (Florence Briggs Thayer)"
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        assert response.status_code == 200, f"API call failed with status code {response.status_code}"

        response_data = response.json()
        assert "prediction" in response_data, "Missing prediction field"
        assert "prediction_label" in response_data, "Missing prediction_label field"
        assert "probability_survived" in response_data, "Missing probability_survived field"
        assert "passenger_details" in response_data, "Missing passenger_details field"

        assert isinstance(response_data["prediction"], int), "Prediction should be an integer"
        assert response_data["prediction_label"] == "Survived", f"Expected 'Survived', got {response_data['prediction_label']}"
        assert response_data["prediction"] == 1, f"Expected prediction 1, got {response_data['prediction']}"
        
        print(" Survive prediction test passed")
        return True
    except Exception as e:
        print(f" Survive prediction test failed: {e}")
        return False

def test_predict_endpoint_dies():
    """Test prediction for a passenger who should not survive."""
    payload = {
        "Pclass": 3,
        "Age": 35.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 8.05,
        "Sex": "male",
        "Embarked": "S",
        "Name": "Heikkinen, Mr. Lauri"
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        assert response.status_code == 200, f"API call failed with status code {response.status_code}"

        response_data = response.json()
        assert "prediction" in response_data, "Missing prediction field"
        assert "prediction_label" in response_data, "Missing prediction_label field"
        assert "probability_survived" in response_data, "Missing probability_survived field"
        assert "passenger_details" in response_data, "Missing passenger_details field"

        assert isinstance(response_data["prediction"], int), "Prediction should be an integer"
        assert response_data["prediction_label"] == "Did not survive", f"Expected 'Did not survive', got {response_data['prediction_label']}"
        assert response_data["prediction"] == 0, f"Expected prediction 0, got {response_data['prediction']}"
        
        print(" Non-survive prediction test passed")
        return True
    except Exception as e:
        print(f" Non-survive prediction test failed: {e}")
        return False

def test_batch_predict():
    """Test batch prediction endpoint."""
    payload = {
        "passengers": [
            {
                "Pclass": 1,
                "Age": 38.0,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 71.2833,
                "Sex": "female",
                "Embarked": "C",
                "Name": "Cumings, Mrs. John Bradley (Florence Briggs Thayer)"
            },
            {
                "Pclass": 3,
                "Age": 35.0,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 8.05,
                "Sex": "male",
                "Embarked": "S",
                "Name": "Heikkinen, Mr. Lauri"
            }
        ]
    }

    try:
        response = requests.post(f"{API_URL}/batch_predict", json=payload)
        assert response.status_code == 200, f"Batch API call failed with status code {response.status_code}"

        response_data = response.json()
        assert "predictions" in response_data, "Missing predictions field"
        assert len(response_data["predictions"]) == 2, f"Expected 2 predictions, got {len(response_data['predictions'])}"
        
        # Check first prediction (should survive)
        assert response_data["predictions"][0]["prediction"] == 1, "First passenger should survive"
        assert response_data["predictions"][0]["prediction_label"] == "Survived", "First passenger should be labeled as survived"
        
        # Check second prediction (should not survive)
        assert response_data["predictions"][1]["prediction"] == 0, "Second passenger should not survive"
        assert response_data["predictions"][1]["prediction_label"] == "Did not survive", "Second passenger should be labeled as did not survive"
        
        print(" Batch prediction test passed")
        return True
    except Exception as e:
        print(f" Batch prediction test failed: {e}")
        return False

if __name__ == "__main__":
    print(f"Testing API at {API_URL}")
    print("=" * 50)
    
    tests = [
        test_health_endpoint,
        test_predict_endpoint_survives,
        test_predict_endpoint_dies,
        test_batch_predict
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("=" * 50)
    if all(results):
        print("🎉 All tests passed!")
    else:
        print(f"❌ Some tests failed. Passed: {sum(results)}/{len(results)}")
        exit(1)