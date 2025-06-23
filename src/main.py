import os
import shutil
import uuid
import pandas as pd
import numpy as np # Added for np.full and np.array
from fastapi import FastAPI, File, UploadFile, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader, APIKey
from pydantic import BaseModel
from typing import List, Dict
import tensorflow as tf
from tensorflow.keras.models import load_model

# Assuming train.py and model.py are in the same src directory
from .train import load_and_preprocess_data, train_model, save_mappings, load_mappings
# create_ncf_model is not directly used by main.py for inference if model is loaded
# from .model import create_ncf_model # Keep this commented unless needed by main directly

# --- Configuration & Globals ---
API_KEY_NAME = "X-API-Key"
MODELS_BASE_DIR = "models_store" # Directory to store models and mappings

# In-memory store for API keys and their associated model paths
# In a real application, this would be a database (e.g., PostgreSQL, Redis)
# Structure: { 'api_key_string': {'model_path': 'path', 'user_map_path': 'path', 'item_map_path': 'path', 'num_users': N, 'num_items': M} }
API_KEYS_DB: Dict[str, Dict] = {
    "testkey123": { # Example pre-existing key, perhaps for a pre-trained model
        "model_path": os.path.join(MODELS_BASE_DIR, "testkey123", "ncf_model.h5"),
        "mappings_path": os.path.join(MODELS_BASE_DIR, "testkey123", "ncf_mappings.json"),
        # num_users/items would be populated if a model was actually there
    }
}

os.makedirs(MODELS_BASE_DIR, exist_ok=True)

app = FastAPI(title="E-commerce Recommendation System API", version="0.1.0")

# --- API Key Authentication ---
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(key: str = Security(api_key_header)):
    if key in API_KEYS_DB:
        return key
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# --- Pydantic Models for Request/Response ---
class TrainResponse(BaseModel):
    message: str
    api_key: str
    model_path: str
    mappings_path: str

class RecommendationRequest(BaseModel):
    user_id: str # Original user_id string
    count: int = 10

class RecommendationItem(BaseModel):
    item_id: str # Original item_id string
    score: float

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    user_id: str

# --- Helper Functions ---
def get_model_and_mappings_for_key(api_key: str):
    if api_key not in API_KEYS_DB or \
       not os.path.exists(API_KEYS_DB[api_key].get("model_path", "")) or \
       not os.path.exists(API_KEYS_DB[api_key].get("mappings_path", "")):
        raise HTTPException(status_code=404, detail="Model or mappings not found for this API key. Ensure the model is trained or paths are correct.")

    model_data = API_KEYS_DB[api_key]
    try:
        # Check if files exist before loading
        if not os.path.exists(model_data["model_path"]):
            raise HTTPException(status_code=404, detail=f"Model file not found at {model_data['model_path']}")
        if not os.path.exists(model_data["mappings_path"]):
            raise HTTPException(status_code=404, detail=f"Mappings file not found at {model_data['mappings_path']}")

        model = load_model(model_data["model_path"])
        user_map, item_map = load_mappings(model_data["mappings_path"])
        # Invert maps for easy lookup from index to original ID
        idx_to_item_map = {idx: item_id for item_id, idx in item_map.items()}
        return model, user_map, item_map, idx_to_item_map, model_data.get("num_users"), model_data.get("num_items")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/mappings: {str(e)}")

# --- API Endpoints ---
@app.post("/v1/train", response_model=TrainResponse)
async def train_new_model(training_data: UploadFile = File(...)):
    """
    Uploads a CSV dataset, trains a new NCF model, and returns an API key for it.
    The CSV should contain 'user_id', 'item_id', and optionally 'interaction_score'.
    """
    temp_file_path = None
    new_api_key_generated = None # To track if API key was generated for cleanup
    try:
        new_api_key = str(uuid.uuid4())
        new_api_key_generated = new_api_key # Mark as generated

        model_dir = os.path.join(MODELS_BASE_DIR, new_api_key)
        os.makedirs(model_dir, exist_ok=True)

        model_save_path = os.path.join(model_dir, "ncf_model.h5")
        mappings_save_path = os.path.join(model_dir, "ncf_mappings.json")

        # Save uploaded file temporarily
        temp_file_path = f"temp_{new_api_key}_{training_data.filename}" # Ensure unique temp file name
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(training_data.file, buffer)

        print(f"Training data saved to {temp_file_path}")

        # 1. Load and Preprocess Data
        df_processed, user_map, item_map, num_users, num_items = load_and_preprocess_data(temp_file_path)

        if df_processed.empty or num_users == 0 or num_items == 0:
            raise HTTPException(status_code=400, detail="Processed data is empty or no users/items found. Check data format and content.")

        # 2. Train Model
        trained_model, training_history = train_model(
            df_processed, num_users, num_items,
            model_save_path=model_save_path
        )

        # 3. Save Mappings
        save_mappings(user_map, item_map, mappings_save_path)

        API_KEYS_DB[new_api_key] = {
            "model_path": model_save_path,
            "mappings_path": mappings_save_path,
            "num_users": num_users,
            "num_items": num_items
        }

        return TrainResponse(
            message="Model training initiated and completed successfully.",
            api_key=new_api_key,
            model_path=model_save_path,
            mappings_path=mappings_save_path
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        if new_api_key_generated and os.path.exists(os.path.join(MODELS_BASE_DIR, new_api_key_generated)):
             shutil.rmtree(os.path.join(MODELS_BASE_DIR, new_api_key_generated), ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during training: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if training_data:
            await training_data.close() # Use await for async file close


@app.post("/v1/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    api_key: APIKey = Depends(get_api_key)
):
    """
    Generates item recommendations for a given user_id.
    Requires a valid API key associated with a trained model.
    """
    try:
        model, user_map, item_map, idx_to_item_map, num_users, num_items = get_model_and_mappings_for_key(api_key)

        if request.user_id not in user_map:
            raise HTTPException(status_code=404, detail=f"User ID '{request.user_id}' not found in the model's user mapping. This user was not present in the training data.")

        user_idx = user_map[request.user_id]

        all_item_indices = np.array(list(item_map.values()))

        if num_items is None or num_items == 0: # num_items should be available from API_KEYS_DB
             raise HTTPException(status_code=500, detail="Number of items not available for model, cannot generate candidates.")

        # For NCF, we usually predict for items the user hasn't interacted with.
        # Here, we'll predict for all items for simplicity, as creating the "non-interacted" list
        # requires knowing all positive interactions for the user from the training set.
        candidate_item_indices = all_item_indices

        if candidate_item_indices.size == 0:
             raise HTTPException(status_code=404, detail="No candidate items found for recommendation (item map is empty).")

        user_array = np.full(len(candidate_item_indices), user_idx)

        predictions = model.predict([user_array, candidate_item_indices], batch_size=512)

        results = []
        for i, item_idx_val in enumerate(candidate_item_indices): # Renamed item_idx to item_idx_val
            original_item_id = idx_to_item_map.get(item_idx_val)
            if original_item_id:
                 results.append({"item_id": original_item_id, "score": float(predictions[i][0])})

        results.sort(key=lambda x: x["score"], reverse=True)
        top_n_recommendations = results[:request.count]

        return RecommendationResponse(
            recommendations=[RecommendationItem(**item) for item in top_n_recommendations],
            user_id=request.user_id
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"An unexpected error occurred during recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during recommendation: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Welcome to the E-commerce Recommendation System API. See /docs for API details."}

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server with Uvicorn...")

    # Setup for 'testkey123' for local development demonstration
    if "testkey123" in API_KEYS_DB:
        _model_info = API_KEYS_DB["testkey123"]
        _dummy_model_dir = os.path.dirname(_model_info["model_path"]) # Get dir from path
        os.makedirs(_dummy_model_dir, exist_ok=True)
        _dummy_model_path = _model_info["model_path"]
        _dummy_mappings_path = _model_info["mappings_path"]

        # Check if either model or mappings are missing to create dummies
        if not os.path.exists(_dummy_model_path) or not os.path.exists(_dummy_mappings_path):
            print(f"Creating dummy model/mappings for 'testkey123' in {_dummy_model_dir}...")
            try:
                dummy_users_count = 10
                dummy_items_count = 5

                # Create a placeholder for the model file if it doesn't exist
                if not os.path.exists(_dummy_model_path):
                    # For TensorFlow/Keras, a .h5 file is not just text.
                    # A minimal valid HDF5 file is complex to create from scratch.
                    # So, we'll create a dummy model using Keras and save it.
                    # This requires model.py to be importable.
                    try:
                        from .model import create_ncf_model # Relative import
                        temp_model = create_ncf_model(dummy_users_count, dummy_items_count, embedding_dim=4) # minimal
                        temp_model.save(_dummy_model_path)
                        print(f"Dummy Keras model saved to {_dummy_model_path}")
                    except Exception as model_ex:
                        print(f"Could not create full dummy Keras model due to: {model_ex}. Creating placeholder file.")
                        with open(_dummy_model_path, 'w') as f: f.write("dummy model placeholder - not a valid Keras model")


                # Create dummy mappings if they don't exist
                if not os.path.exists(_dummy_mappings_path):
                    dummy_user_map = {f"user{i}": i for i in range(dummy_users_count)}
                    dummy_item_map = {f"item{j}": j for j in range(dummy_items_count)}
                    save_mappings(dummy_user_map, dummy_item_map, _dummy_mappings_path)
                    print(f"Dummy mappings saved to {_dummy_mappings_path}")

                # Update API_KEYS_DB with counts for the dummy model
                API_KEYS_DB["testkey123"]["num_users"] = dummy_users_count
                API_KEYS_DB["testkey123"]["num_items"] = dummy_items_count
                print(f"Dummy model and mappings for 'testkey123' setup in {_dummy_model_dir}.")
            except Exception as e:
                print(f"Error creating dummy model/mappings for testkey123: {e}")
        else:
            # If files exist, ensure num_users/num_items are in API_KEYS_DB for testkey123
            if "num_users" not in API_KEYS_DB["testkey123"] or "num_items" not in API_KEYS_DB["testkey123"]:
                try:
                    _, item_map_loaded = load_mappings(_dummy_mappings_path) # Corrected to get both maps
                    user_map_loaded, _ = load_mappings(_dummy_mappings_path) # Load user_map as well
                    API_KEYS_DB["testkey123"]["num_users"] = len(user_map_loaded)
                    API_KEYS_DB["testkey123"]["num_items"] = len(item_map_loaded)
                    print(f"Loaded counts for existing 'testkey123' model: {len(user_map_loaded)} users, {len(item_map_loaded)} items.")
                except Exception as e:
                    print(f"Could not load num_users/num_items for existing testkey123: {e}")
            else:
                 print(f"Dummy model/mappings for 'testkey123' already exist and configured at {_dummy_model_dir}.")


    uvicorn.run(app, host="0.0.0.0", port=8000)
