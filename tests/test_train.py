import pytest
import pandas as pd
import numpy as np
import os
import json
import tensorflow as tf # Added for loading saved model
from src.train import load_and_preprocess_data, save_mappings, load_mappings, train_model # Adjusted import path
from src.model import create_ncf_model # For creating a dummy model for train_model test

# Fixture for creating a dummy CSV file for testing
@pytest.fixture(scope="module") # module scope: run once per test module
def dummy_csv_path(tmpdir_factory):
    data = {
        'user_id': ['u1', 'u1', 'u2', 'u2', 'u3', 'u1', 'u4', 'u5', 'u2', 'u3'],
        'item_id': ['i1', 'i2', 'i1', 'i3', 'i2', 'i3', 'i4', 'i5', 'i2', 'i1'],
        'interaction_score': [1, 1, 0, 1, 1, 1, 0, 1, 1, 1] # some implicit negatives
    }
    df = pd.DataFrame(data)
    file_path = tmpdir_factory.mktemp("data").join("dummy_interactions.csv")
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_load_and_preprocess_data(dummy_csv_path):
    df_processed, user_map, item_map, num_users, num_items = load_and_preprocess_data(dummy_csv_path, negative_samples=2)

    assert not df_processed.empty
    assert 'user_idx' in df_processed.columns
    assert 'item_idx' in df_processed.columns
    assert 'label' in df_processed.columns

    assert len(user_map) == num_users
    assert len(item_map) == num_items
    assert num_users == 5 # u1, u2, u3, u4, u5
    assert num_items == 5 # i1, i2, i3, i4, i5

    # Check for positive and negative samples
    assert 1 in df_processed['label'].unique() # Positive samples
    assert 0 in df_processed['label'].unique() # Negative samples

    # Check if mappings are correct
    assert user_map['u1'] == 0 or user_map['u1'] != user_map['u2'] # Basic check

def test_save_and_load_mappings(tmpdir):
    user_map_orig = {'u1': 0, 'u2': 1}
    item_map_orig = {'i1': 0, 'i2': 1}
    file_path = os.path.join(str(tmpdir), "test_mappings.json")

    save_mappings(user_map_orig, item_map_orig, file_path)
    assert os.path.exists(file_path)

    user_map_loaded, item_map_loaded = load_mappings(file_path)
    assert user_map_loaded == user_map_orig
    assert item_map_loaded == item_map_orig

# More involved test for train_model - might be slow, consider marking as integration if too slow
def test_train_model_runs_and_saves(dummy_csv_path, tmpdir):
    df_processed, user_map, item_map, num_users, num_items = load_and_preprocess_data(dummy_csv_path)

    model_save_path = os.path.join(str(tmpdir), "test_model.h5")
    # Mappings are saved separately by the calling logic in train.py's main, not directly by train_model

    # Ensure num_users and num_items are greater than 0 before proceeding
    if num_users == 0 or num_items == 0:
        pytest.skip("Skipping test_train_model_runs_and_saves as no users or items were found in dummy data.")

    trained_model, history = train_model(
        df_processed, num_users, num_items,
        model_save_path=model_save_path,
        epochs=1, # Keep epochs very low for testing speed
        batch_size=32
    )

    assert os.path.exists(model_save_path)
    assert trained_model is not None
    assert 'accuracy' in history.history
    assert 'loss' in history.history

    # Try loading the saved model
    try:
        loaded_model = tf.keras.models.load_model(model_save_path)
        assert loaded_model is not None
    except Exception as e:
        pytest.fail(f"Failed to load saved model: {e}")
