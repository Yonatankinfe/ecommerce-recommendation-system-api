import pytest
import os
import shutil
import pandas as pd
from httpx import AsyncClient
from fastapi.testclient import TestClient # Can use this for sync tests if preferred

# Import the FastAPI app instance
# Assuming your FastAPI app instance is named 'app' in 'src.main'
from src.main import app, API_KEYS_DB, MODELS_BASE_DIR

# Use FastAPI's TestClient for synchronous testing.
# pytestmark = pytest.mark.asyncio # Not needed for TestClient

# Fixture for the TestClient
@pytest.fixture(scope="function")
def client(): # Synchronous fixture
    with TestClient(app) as c: # base_url is implicitly handled by TestClient(app)
        yield c

# Fixture to create a dummy CSV for uploading during tests
@pytest.fixture(scope="module")
def dummy_files_fixture(tmpdir_factory): # Renamed fixture
    data = {
        'user_id': ['u1', 'u1', 'u2', 'u2', 'u3', 'u4', 'u5', 'u1', 'u2', 'u3', 'u4', 'u5'],
        'item_id': ['i1', 'i2', 'i1', 'i3', 'i4', 'i5', 'i1', 'i3', 'i2', 'i2', 'i4', 'i3'],
        'interaction_score': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # All positive for simplicity
    }
    df = pd.DataFrame(data)
    # Use tmpdir_factory for module-scoped temporary directory
    temp_dir = tmpdir_factory.mktemp("data")
    file_path = temp_dir.join("test_train_data.csv")
    df.to_csv(file_path, index=False)

    # Create an empty CSV
    empty_file_path = temp_dir.join("empty.csv")
    open(empty_file_path, 'w').close()

    # Create a non-CSV file (e.g., binary or just text)
    non_csv_file_path = temp_dir.join("not_a_csv.txt")
    with open(non_csv_file_path, 'w') as f:
        f.write("This is not a CSV file.")

    # Create CSV with missing user_id column
    missing_user_data = {'item_id': ['i1', 'i2'], 'interaction_score': [1,0]}
    missing_user_df = pd.DataFrame(missing_user_data)
    missing_user_path = temp_dir.join("missing_user.csv")
    missing_user_df.to_csv(missing_user_path, index=False)

    # Create CSV with missing item_id column
    missing_item_data = {'user_id': ['u1', 'u2'], 'interaction_score': [1,0]}
    missing_item_df = pd.DataFrame(missing_item_data)
    missing_item_path = temp_dir.join("missing_item.csv")
    missing_item_df.to_csv(missing_item_path, index=False)


    return {
        "valid": str(file_path),
        "empty": str(empty_file_path),
        "non_csv": str(non_csv_file_path),
        "missing_user": str(missing_user_path),
        "missing_item": str(missing_item_path),
    }

# Cleanup MODELS_BASE_DIR after tests run
@pytest.fixture(scope="session", autouse=True)
def cleanup_model_store():
    # Store the initial state of API_KEYS_DB related to 'testkey123' if it exists
    initial_testkey_state = API_KEYS_DB.get("testkey123")

    yield
    # This runs after all tests in the session
    if os.path.exists(MODELS_BASE_DIR):
        print(f"Cleaning up {MODELS_BASE_DIR}...")
        # Be more selective: remove only directories that are UUIDs (trained models)
        # or the specific "testkey123" if it was modified/created by a test directly.
        # This avoids deleting a potentially valid pre-existing "testkey123" setup from main.py
        for item in os.listdir(MODELS_BASE_DIR):
            item_path = os.path.join(MODELS_BASE_DIR, item)
            if os.path.isdir(item_path):
                try:
                    # Attempt to parse as UUID to identify test-generated model dirs
                    from uuid import UUID
                    UUID(item, version=4)
                    shutil.rmtree(item_path, ignore_errors=True)
                    print(f"Removed test-generated model directory: {item_path}")
                except ValueError:
                    # If not a UUID, it might be "testkey123" or other.
                    # Only remove "testkey123" if it's not the one from initial_testkey_state
                    # or if tests specifically created it. For simplicity here, we'll just log.
                    # A more robust way would be to tag test-generated dirs.
                    print(f"Skipping non-UUID directory during cleanup: {item_path}")

    # Clean up API_KEYS_DB for keys generated during tests (UUIDs)
    keys_to_delete = [key for key in API_KEYS_DB if key != "testkey123"] # Keep testkey123
    for key in keys_to_delete:
        del API_KEYS_DB[key]

    # Restore initial 'testkey123' if it was present, as main.py might set it up on startup
    # This part is tricky because main.py's startup logic might run again if client is re-initialized.
    # For now, if 'testkey123' was modified, this doesn't revert the files, only the DB entry.
    # The dummy model creation in main.py should ideally handle idempotent creation.
    if initial_testkey_state and "testkey123" not in API_KEYS_DB:
         API_KEYS_DB["testkey123"] = initial_testkey_state
    elif not initial_testkey_state and "testkey123" in API_KEYS_DB:
        # If main.py added it but it wasn't there initially, remove it.
        # This scenario is less likely if main.py's setup is consistent.
        # del API_KEYS_DB["testkey123"]
        pass # Let main.py handle its "testkey123" setup on next app load if needed.


def test_root_endpoint(client: TestClient): # Sync, TestClient
    response = client.get("/") # No await
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the E-commerce Recommendation System API. See /docs for API details."}

def test_train_endpoint_success(client: TestClient, dummy_files_fixture): # Sync, TestClient
    with open(dummy_files_fixture["valid"], 'rb') as f:
        files = {'training_data': ('test_train_data.csv', f, 'text/csv')}
        response = client.post("/v1/train", files=files) # No await

    assert response.status_code == 200
    data = response.json()
    assert "api_key" in data
    assert "model_path" in data
    assert "mappings_path" in data
    assert data["message"] == "Model training initiated and completed successfully."

    api_key = data["api_key"]
    assert api_key in API_KEYS_DB
    assert os.path.exists(data["model_path"])
    assert os.path.exists(data["mappings_path"])
    assert os.path.exists(API_KEYS_DB[api_key]["model_path"])
    assert os.path.exists(API_KEYS_DB[api_key]["mappings_path"])

def test_train_endpoint_empty_csv(client: TestClient, dummy_files_fixture):
    with open(dummy_files_fixture["empty"], 'rb') as f:
        files = {'training_data': ('empty.csv', f, 'text/csv')}
        response = client.post("/v1/train", files=files)
    # Based on current train.py, this will lead to num_users=0, num_items=0
    # which then leads to "Processed data is empty or no users/items found"
    # Update: main.py's general exception handler catches pd.errors.EmptyDataError from load_and_preprocess_data
    # and returns a 500.
    assert response.status_code == 500 # Internal Server Error due to unhandled pandas EmptyDataError
    detail = response.json().get("detail", "").lower()
    assert "error occurred during training" in detail # Generic message from main.py
    assert "no columns to parse from file" in detail # Specific pandas error propagated

def test_train_endpoint_malformed_csv(client: TestClient, dummy_files_fixture):
    with open(dummy_files_fixture["non_csv"], 'rb') as f:
        files = {'training_data': ('not_a_csv.txt', f, 'text/csv')} # Content-Type is still csv
        response = client.post("/v1/train", files=files)
    # This should fail during pd.read_csv() in load_and_preprocess_data
    assert response.status_code == 500 # Or 400 if pandas error is caught specifically
    # The specific error message might vary depending on pandas version and error handling in train.py
    # For now, check for a generic server error or a more specific parsing error if possible.
    # Example: "An error occurred during training: Error tokenizing data."
    assert "error occurred during training" in response.json().get("detail", "").lower()

def test_train_endpoint_missing_user_id_column(client: TestClient, dummy_files_fixture):
    with open(dummy_files_fixture["missing_user"], 'rb') as f:
        files = {'training_data': ('missing_user.csv', f, 'text/csv')}
        response = client.post("/v1/train", files=files)
    assert response.status_code == 500 # load_and_preprocess_data raises KeyError, caught by general Exception
    assert "error occurred during training" in response.json().get("detail", "").lower()
    # To be more specific, we'd need to ensure KeyError leads to a 4xx error in main.py
    # Currently, it's a general 500: "An error occurred during training: 'user_id'"

def test_train_endpoint_missing_item_id_column(client: TestClient, dummy_files_fixture):
    with open(dummy_files_fixture["missing_item"], 'rb') as f:
        files = {'training_data': ('missing_item.csv', f, 'text/csv')}
        response = client.post("/v1/train", files=files)
    assert response.status_code == 500 # Similar to missing user_id
    assert "error occurred during training" in response.json().get("detail", "").lower()

def test_recommendations_endpoint_no_auth(client: TestClient):
    response = client.post("/v1/recommendations", json={"user_id": "u1", "count": 5})
    assert response.status_code == 403
    detail = response.json().get("detail", "").lower()
    assert "not authenticated" in detail or "could not validate credentials" in detail

def test_recommendations_endpoint_bad_auth(client: TestClient):
    response = client.post(
        "/v1/recommendations",
        json={"user_id": "u1", "count": 5},
        headers={"X-API-Key": "fakekey"}
    )
    assert response.status_code == 403
    assert "Could not validate credentials" in response.json().get("detail", "")

def test_recommendations_endpoint_with_trained_model_success(client: TestClient, dummy_files_fixture):
    api_key_to_use = None

    with open(dummy_files_fixture["valid"], 'rb') as f:
        files = {'training_data': ('test_train_data.csv', f, 'text/csv')}
        train_response = client.post("/v1/train", files=files)

    assert train_response.status_code == 200, f"Training failed: {train_response.text}"
    train_data = train_response.json()
    api_key_to_use = train_data["api_key"]

    assert api_key_to_use is not None

    # Ensure user 'u1' is in the valid training data
    rec_response = client.post(
        "/v1/recommendations",
        json={"user_id": "u1", "count": 3},
        headers={"X-API-Key": api_key_to_use}
    )

    assert rec_response.status_code == 200, f"Recommendations failed: {rec_response.text}"
    rec_data = rec_response.json()
    assert "recommendations" in rec_data
    assert "user_id" in rec_data
    assert rec_data["user_id"] == "u1"
    assert len(rec_data["recommendations"]) <= 3
    if rec_data["recommendations"]:
        assert "item_id" in rec_data["recommendations"][0]
        assert "score" in rec_data["recommendations"][0]

def test_recommendations_missing_model_file(client: TestClient, dummy_files_fixture):
    # 1. Train a model
    with open(dummy_files_fixture["valid"], 'rb') as f:
        files = {'training_data': ('test_train_data.csv', f, 'text/csv')}
        train_response = client.post("/v1/train", files=files)
    assert train_response.status_code == 200
    train_data = train_response.json()
    api_key = train_data["api_key"]
    model_path = train_data["model_path"]

    # 2. Delete the model file
    if os.path.exists(model_path):
        os.remove(model_path)

    # 3. Try to get recommendations
    response = client.post(
        "/v1/recommendations",
        json={"user_id": "u1", "count": 5},
        headers={"X-API-Key": api_key}
    )
    assert response.status_code == 404 # Model file not found
    expected_detail_substring = "model or mappings not found for this api key"
    assert expected_detail_substring in response.json().get("detail", "").lower()

def test_recommendations_missing_mappings_file(client: TestClient, dummy_files_fixture):
    # 1. Train a model
    with open(dummy_files_fixture["valid"], 'rb') as f:
        files = {'training_data': ('test_train_data.csv', f, 'text/csv')}
        train_response = client.post("/v1/train", files=files)
    assert train_response.status_code == 200
    train_data = train_response.json()
    api_key = train_data["api_key"]
    mappings_path = train_data["mappings_path"]

    # 2. Delete the mappings file
    if os.path.exists(mappings_path):
        os.remove(mappings_path)

    # 3. Try to get recommendations
    response = client.post(
        "/v1/recommendations",
        json={"user_id": "u1", "count": 5},
        headers={"X-API-Key": api_key}
    )
    assert response.status_code == 404 # Mappings file not found
    expected_detail_substring = "model or mappings not found for this api key"
    assert expected_detail_substring in response.json().get("detail", "").lower()

def test_recommendations_corrupted_mappings_file(client: TestClient, dummy_files_fixture):
    # 1. Train a model
    with open(dummy_files_fixture["valid"], 'rb') as f:
        files = {'training_data': ('test_train_data.csv', f, 'text/csv')}
        train_response = client.post("/v1/train", files=files)
    assert train_response.status_code == 200
    train_data = train_response.json()
    api_key = train_data["api_key"]
    mappings_path = train_data["mappings_path"]

    # 2. Corrupt the mappings file (write invalid JSON)
    with open(mappings_path, 'w') as f:
        f.write("this is not valid json")

    # 3. Try to get recommendations
    response = client.post(
        "/v1/recommendations",
        json={"user_id": "u1", "count": 5},
        headers={"X-API-Key": api_key}
    )
    assert response.status_code == 500 # Error loading mappings
    assert "error loading model/mappings" in response.json().get("detail", "").lower()

def test_recommendation_for_unknown_user(client: TestClient, dummy_files_fixture):
    with open(dummy_files_fixture["valid"], 'rb') as f:
        files = {'training_data': ('test_train_data.csv', f, 'text/csv')}
        train_response = client.post("/v1/train", files=files)
    assert train_response.status_code == 200, f"Training failed: {train_response.text}"
    api_key = train_response.json()["api_key"]

    response = client.post(
        "/v1/recommendations",
        json={"user_id": "unknown_user_id_qwerty", "count": 5},
        headers={"X-API-Key": api_key}
    )
    assert response.status_code == 404
    assert "not found in the model's user mapping" in response.json().get("detail", "")

def test_recommendation_with_testkey123_if_model_exists(client: TestClient):
    # This test relies on the dummy model setup for 'testkey123' in main.py
    # It checks if the pre-configured 'testkey123' can return recommendations.
    # This test might fail if the dummy model in main.py is just a placeholder file
    # and not a real, loadable Keras model.

    test_key_info = API_KEYS_DB.get("testkey123")
    if not test_key_info or \
       not test_key_info.get("model_path") or \
       not test_key_info.get("mappings_path") or \
       not os.path.exists(test_key_info["model_path"]) or \
       not os.path.exists(test_key_info["mappings_path"]):
        pytest.skip("testkey123 model or mappings not found or not configured. Skipping this test.")

    # Check if the model is a placeholder (hacky check based on main.py's dummy placeholder)
    # A more robust check would be to try loading it here, but that adds overhead.
    if os.path.getsize(test_key_info["model_path"]) < 1024: # Arbitrary small size for placeholder
         try:
             # Attempt to load to confirm it's a real model
             from tensorflow.keras.models import load_model
             load_model(test_key_info["model_path"])
         except Exception as e:
             pytest.skip(f"testkey123 model at {test_key_info['model_path']} is likely a placeholder or invalid: {e}. Skipping.")


    # Assuming 'user0' or similar is in the dummy mappings for 'testkey123'
    # The dummy mappings in main.py are like user_map = {f"user{i}": i for i in range(dummy_users_count)}
    user_to_test = "user0"
    if "num_users" in test_key_info and test_key_info["num_users"] == 0:
         pytest.skip("testkey123 is configured but has 0 users. Skipping.")
    if "num_users" not in test_key_info: # If num_users is not set, we can't be sure user0 exists
         print("Warning: num_users not set for testkey123. Assuming user0 exists for test.")


    response = client.post( # Removed await
        "/v1/recommendations",
        json={"user_id": user_to_test, "count": 2},
        headers={"X-API-Key": "testkey123"}
    )

    # If the dummy model is just a placeholder, this will likely fail at model loading (500)
    # or if user_to_test is not in its dummy mappings (404).
    assert response.status_code == 200, f"Response: {response.text}"
    data = response.json()
    assert "recommendations" in data
    assert data["user_id"] == user_to_test
    assert len(data["recommendations"]) <= 2
