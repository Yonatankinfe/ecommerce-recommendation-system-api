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
    assert num_users == 4 # u1, u2, u3, u5 (u4 only has 0-score interactions)
    assert num_items == 4 # i1, i2, i3, i5 (i4 only has 0-score interactions)

    # Check for positive and negative samples
    assert 1 in df_processed['label'].unique() # Positive samples
    assert 0 in df_processed['label'].unique() # Negative samples

    # Check if mappings are correct
    assert user_map['u1'] == 0 or user_map['u1'] != user_map['u2'] # Basic check

# Fixture for creating an empty CSV file
@pytest.fixture
def empty_csv_path(tmpdir):
    file_path = tmpdir.join("empty.csv")
    # Create an empty file, or a file with only headers if that's more appropriate
    # For pd.read_csv, an empty file is fine. If headers are needed:
    # pd.DataFrame(columns=['user_id', 'item_id', 'interaction_score']).to_csv(file_path, index=False)
    open(file_path, 'a').close()
    return str(file_path)

# Fixture for CSV missing user_id
@pytest.fixture
def missing_user_id_csv_path(tmpdir):
    data = {'item_id': ['i1', 'i2'], 'interaction_score': [1, 0]}
    df = pd.DataFrame(data)
    file_path = tmpdir.join("missing_user.csv")
    df.to_csv(file_path, index=False)
    return str(file_path)

# Fixture for CSV missing item_id
@pytest.fixture
def missing_item_id_csv_path(tmpdir):
    data = {'user_id': ['u1', 'u2'], 'interaction_score': [1, 0]}
    df = pd.DataFrame(data)
    file_path = tmpdir.join("missing_item.csv")
    df.to_csv(file_path, index=False)
    return str(file_path)

# Fixture for a user who has interacted with all items
@pytest.fixture
def user_interacted_all_csv_path(tmpdir):
    data = { # u1 interacts with i1, i2. u2 interacts with i1. Total items i1, i2.
        'user_id': ['u1', 'u1', 'u2'],
        'item_id': ['i1', 'i2', 'i1'],
        'interaction_score': [1, 1, 1]
    }
    df = pd.DataFrame(data)
    file_path = tmpdir.join("user_interacted_all.csv")
    df.to_csv(file_path, index=False)
    return str(file_path)


def test_load_data_empty_csv(empty_csv_path):
    # Expecting pd.errors.EmptyDataError or similar if pandas can't read it,
    # or specific handling if the function is designed for it.
    # Current implementation might raise an error or return empty structures.
    # Let's assume it should handle it by returning empty/zero values.
    # Update: The function currently propagates pandas.errors.EmptyDataError
    with pytest.raises(pd.errors.EmptyDataError):
        load_and_preprocess_data(empty_csv_path)

def test_load_data_missing_user_id_column(missing_user_id_csv_path):
    # This should ideally raise a KeyError or ValueError if 'user_id' is essential.
    with pytest.raises(KeyError): # Or whatever error the function is designed to raise
        load_and_preprocess_data(missing_user_id_csv_path)

def test_load_data_missing_item_id_column(missing_item_id_csv_path):
    # Similar to missing user_id
    with pytest.raises(KeyError): # Or whatever error the function is designed to raise
        load_and_preprocess_data(missing_item_id_csv_path)

def test_load_data_user_interacted_with_all_items(user_interacted_all_csv_path):
    # For user 'u1' who interacted with all items ('i1', 'i2'), no negative samples should be generated for 'u1'.
    # For user 'u2' who interacted with 'i1', negative sample 'i2' can be generated.
    df_processed, user_map, item_map, num_users, num_items = load_and_preprocess_data(user_interacted_all_csv_path, negative_samples=1)

    assert num_users == 2 # u1, u2
    assert num_items == 2 # i1, i2

    u1_idx = user_map['u1']
    # u2_idx = user_map['u2']
    # i1_idx = item_map['i1']
    # i2_idx = item_map['i2']

    # Check that u1 has only positive interactions (label 1) and no negative samples (label 0)
    u1_interactions = df_processed[df_processed['user_idx'] == u1_idx]
    assert all(u1_interactions['label'] == 1)
    # Original interactions for u1 were (u1,i1) and (u1,i2).
    # Since u1 interacted with all items, no negative samples for u1.
    assert len(u1_interactions) == 2 # Should only have the two positive interactions

    # u2 interacted with i1. A negative sample (u2, i2) should be possible.
    u2_interactions = df_processed[df_processed['user_idx'] == user_map['u2']]
    assert any(u2_interactions['label'] == 1) # (u2,i1)
    if num_items > 1 : # only generate negative if there are other items
        assert any(u2_interactions['label'] == 0) # (u2,i2) as negative, if i2 exists and u2 hasn't interacted.

    # Total positive interactions = 3.
    # For u1, 0 negative samples.
    # For u2 (interacted with i1), 1 negative sample (i2) can be generated.
    # Total expected samples = 3 positive + (0 for u1) + (1 for u2) = 4
    # This count depends on how df_positive is constructed (unique user-item pairs or all rows)
    # The current load_and_preprocess_data uses each row in df_positive to generate negatives.
    # u1,i1 (pos) -> no neg for u1
    # u1,i2 (pos) -> no neg for u1
    # u2,i1 (pos) -> neg (u2,i2) generated
    # So, 3 positive rows, 1 negative row. Total 4 rows.
    assert len(df_processed) == 3 + 1 # 3 positive, 1 negative for u2


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
