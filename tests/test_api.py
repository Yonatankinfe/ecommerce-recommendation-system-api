import pytest
import os
import shutil
import pandas as pd
from httpx import AsyncClient
from fastapi.testclient import TestClient # Can use this for sync tests if preferred

# Import the FastAPI app instance
# Assuming your FastAPI app instance is named 'app' in 'src.main'
from src.main import app, API_KEYS_DB, MODELS_BASE_DIR

# Use pytest-asyncio for async tests with httpx
pytestmark = pytest.mark.asyncio


# Fixture for the AsyncClient
@pytest.fixture(scope="module")
async def client():
    async with AsyncClient(app=app, base_url="http://127.0.0.1:8000") as ac: # Base URL must match server
        yield ac

# Fixture to create a dummy CSV for uploading during tests
@pytest.fixture(scope="module")
def dummy_train_csv_path(tmpdir_factory):
    data = {
        'user_id': ['u1', 'u1', 'u2', 'u2', 'u3', 'u4', 'u5', 'u1', 'u2', 'u3', 'u4', 'u5'],
        'item_id': ['i1', 'i2', 'i1', 'i3', 'i4', 'i5', 'i1', 'i3', 'i2', 'i2', 'i4', 'i3'],
        'interaction_score': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # All positive for simplicity
    }
    df = pd.DataFrame(data)
    # Use tmpdir_factory for module-scoped temporary directory
    file_path = tmpdir_factory.mktemp("data").join("test_train_data.csv")
    df.to_csv(file_path, index=False)
    return str(file_path)

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


async def test_root_endpoint(client: AsyncClient):
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the E-commerce Recommendation System API. See /docs for API details."}

async def test_train_endpoint(client: AsyncClient, dummy_train_csv_path):
    with open(dummy_train_csv_path, 'rb') as f:
        files = {'training_data': ('test_train_data.csv', f, 'text/csv')}
        response = await client.post("/v1/train", files=files)

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


async def test_recommendations_endpoint_no_auth(client: AsyncClient):
    response = await client.post("/v1/recommendations", json={"user_id": "u1", "count": 5})
    assert response.status_code == 403
    detail = response.json().get("detail", "").lower()
    assert "not authenticated" in detail or "could not validate credentials" in detail


async def test_recommendations_endpoint_bad_auth(client: AsyncClient):
    response = await client.post(
        "/v1/recommendations",
        json={"user_id": "u1", "count": 5},
        headers={"X-API-Key": "fakekey"}
    )
    assert response.status_code == 403
    assert "Could not validate credentials" in response.json().get("detail", "")


async def test_recommendations_endpoint_with_trained_model(client: AsyncClient, dummy_train_csv_path):
    api_key_to_use = None

    with open(dummy_train_csv_path, 'rb') as f:
        files = {'training_data': ('test_train_data.csv', f, 'text/csv')}
        train_response = await client.post("/v1/train", files=files)

    assert train_response.status_code == 200, f"Training failed: {train_response.text}"
    train_data = train_response.json()
    api_key_to_use = train_data["api_key"]

    assert api_key_to_use is not None

    rec_response = await client.post(
        "/v1/recommendations",
        json={"user_id": "u1", "count": 3}, # u1 is in dummy_train_csv_path
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

async def test_recommendation_for_unknown_user(client: AsyncClient, dummy_train_csv_path):
    with open(dummy_train_csv_path, 'rb') as f:
        files = {'training_data': ('test_train_data.csv', f, 'text/csv')}
        train_response = await client.post("/v1/train", files=files)
    assert train_response.status_code == 200, f"Training failed: {train_response.text}"
    api_key = train_response.json()["api_key"]

    response = await client.post(
        "/v1/recommendations",
        json={"user_id": "unknown_user_id_qwerty", "count": 5},
        headers={"X-API-Key": api_key}
    )
    assert response.status_code == 404
    assert "not found in the model's user mapping" in response.json().get("detail", "")

async def test_recommendation_with_testkey123_if_model_exists(client: AsyncClient):
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


    response = await client.post(
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
