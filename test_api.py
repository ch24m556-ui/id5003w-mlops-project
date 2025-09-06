# test_api.py
import requests

API_URL = "http://127.0.0.1:8000/predict"

def test_predict_endpoint_survives():
    """
    Tests the /predict endpoint with data for a passenger who should survive.
    """
    # Sample data for a passenger with a high chance of survival
    # (e.g., female, high class, moderate fare)
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

    response = requests.post(API_URL, json=payload)

    # 1. Check if the request was successful
    assert response.status_code == 200, f"API call failed with status code {response.status_code}"

    # 2. Check the response format
    response_data = response.json()
    assert "prediction" in response_data
    assert "prediction_label" in response_data
    assert "probability_survived" in response_data

    # 3. Check the prediction value and type
    assert isinstance(response_data["prediction"], int)
    assert response_data["prediction_label"] == "Survived"
    assert response_data["prediction"] == 1
    print("\ntest_predict_endpoint_survives: PASSED")


def test_predict_endpoint_dies():
    """
    Tests the /predict endpoint with data for a passenger who should not survive.
    """
    # Sample data for a passenger with a low chance of survival
    # (e.g., male, low class)
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

    response = requests.post(API_URL, json=payload)

    # 1. Check if the request was successful
    assert response.status_code == 200

    # 2. Check the response format
    response_data = response.json()
    assert "prediction" in response_data
    assert "prediction_label" in response_data
    assert "probability_survived" in response_data

    # 3. Check the prediction value and type
    assert isinstance(response_data["prediction"], int)
    assert response_data["prediction_label"] == "Did not survive"
    assert response_data["prediction"] == 0
    print("test_predict_endpoint_dies: PASSED")

if __name__ == "__main__":
    test_predict_endpoint_survives()
    test_predict_endpoint_dies()
    print("All tests passed!")