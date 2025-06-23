import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import tensorflow as tf # Added for AUC

from .model import create_ncf_model # Assuming model.py is in the same directory or src is in PYTHONPATH

# --- Configuration ---
DEFAULT_EMBEDDING_DIM = 32
DEFAULT_MLP_LAYERS = [64, 32, 16]
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 10 # Low for quick example, should be higher for real training
DEFAULT_NEGATIVE_SAMPLES = 4 # Number of negative samples per positive sample

def load_and_preprocess_data(csv_file_path, negative_samples=DEFAULT_NEGATIVE_SAMPLES):
    """
    Loads data from a CSV file, performs entity mapping, and generates negative samples.

    Args:
        csv_file_path (str): Path to the CSV file.
                               Expected columns: 'user_id', 'item_id',
                                                 'interaction_score' (optional, 1 for positive if not present),
                                                 'timestamp' (optional, not used in this version).
        negative_samples (int): Number of negative samples to generate per positive interaction.

    Returns:
        tuple: Contains:
            - df_processed (pd.DataFrame): Processed dataframe with 'user_idx', 'item_idx', 'label'.
            - user_map (dict): Mapping from original user_id to integer index.
            - item_map (dict): Mapping from original item_id to integer index.
            - num_users (int): Total number of unique users.
            - num_items (int): Total number of unique items.
    """
    print(f"Loading data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)

    # Assume positive interaction if 'interaction_score' is not present or > 0
    if 'interaction_score' in df.columns:
        df['label'] = (df['interaction_score'] > 0).astype(int)
    else:
        df['label'] = 1 # Default to positive interaction

    # Keep only positive interactions for initial user/item mapping and negative sampling base
    df_positive = df[df['label'] == 1].copy()

    # Entity Mapping
    unique_users = df_positive['user_id'].unique()
    unique_items = df_positive['item_id'].unique()

    user_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_map = {item_id: idx for idx, item_id in enumerate(unique_items)}

    num_users = len(unique_users)
    num_items = len(unique_items)

    print(f"Found {num_users} unique users and {num_items} unique items.")

    df_positive['user_idx'] = df_positive['user_id'].map(user_map)
    df_positive['item_idx'] = df_positive['item_id'].map(item_map)

    # --- Negative Sampling ---
    print(f"Generating {negative_samples} negative samples per positive interaction...")
    df_negatives_list = []
    all_item_indices = set(range(num_items))

    for _, row in df_positive.iterrows():
        user_idx = row['user_idx']
        interacted_items = set(df_positive[df_positive['user_idx'] == user_idx]['item_idx'])
        non_interacted_items = list(all_item_indices - interacted_items)

        if not non_interacted_items: # User interacted with all items
            continue

        num_samples_to_generate = min(negative_samples, len(non_interacted_items))
        sampled_negative_items = np.random.choice(non_interacted_items, size=num_samples_to_generate, replace=False)

        for item_idx in sampled_negative_items:
            df_negatives_list.append({'user_idx': user_idx, 'item_idx': item_idx, 'label': 0})

    df_negatives = pd.DataFrame(df_negatives_list)

    # Combine positive and negative samples
    df_processed = pd.concat([df_positive[['user_idx', 'item_idx', 'label']], df_negatives], ignore_index=True)
    df_processed = df_processed.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle

    print(f"Total samples after negative sampling: {len(df_processed)}")
    return df_processed, user_map, item_map, num_users, num_items


def train_model(df_processed, num_users, num_items,
                model_save_path='model.h5', mappings_save_path='mappings.json',
                embedding_dim=DEFAULT_EMBEDDING_DIM, mlp_layers=DEFAULT_MLP_LAYERS,
                batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCHS):
    """
    Trains the NCF model and saves it along with the mappings.
    """
    print("Preparing data for training...")
    X_user = df_processed['user_idx'].values
    X_item = df_processed['item_idx'].values
    y = df_processed['label'].values

    # Train-validation split (optional, but good practice)
    X_user_train, X_user_val, X_item_train, X_item_val, y_train, y_val = train_test_split(
        X_user, X_item, y, test_size=0.2, random_state=42
    )

    print(f"Training with {len(X_user_train)} samples, validating with {len(X_user_val)} samples.")

    model = create_ncf_model(num_users, num_items, embedding_dim, mlp_layers)

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]) # Added AUC

    print("Starting model training...")
    history = model.fit(
        [X_user_train, X_item_train], y_train,
        validation_data=([X_user_val, X_item_val], y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    print(f"Training complete. Saving model to {model_save_path} and mappings to {mappings_save_path}")
    model.save(model_save_path)
    # Mappings are returned by load_and_preprocess_data, should be saved by the caller

    print("Model training process finished.")
    return model, history

def save_mappings(user_map, item_map, file_path):
    mappings = {
        'user_map': user_map,
        'item_map': item_map
    }
    with open(file_path, 'w') as f:
        json.dump(mappings, f)
    print(f"Mappings saved to {file_path}")

def load_mappings(file_path):
    with open(file_path, 'r') as f:
        mappings = json.load(f)
    print(f"Mappings loaded from {file_path}")
    return mappings['user_map'], mappings['item_map']


if __name__ == '__main__':
    print("Starting training script example...")
    # Create a dummy CSV for testing
    dummy_data = {
        'user_id': ['u1', 'u1', 'u2', 'u2', 'u3', 'u3', 'u4', 'u1', 'u2', 'u4', 'u5'],
        'item_id': ['i1', 'i2', 'i1', 'i3', 'i2', 'i4', 'i1', 'i3', 'i4', 'i5', 'i1'],
        'interaction_score': [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], # 0 indicates a negative interaction already present
        'timestamp': [1678886400 + i for i in range(11)]
    }
    dummy_csv_path = 'dummy_interactions.csv'
    pd.DataFrame(dummy_data).to_csv(dummy_csv_path, index=False)
    print(f"Dummy data created at {dummy_csv_path}")

    # --- 1. Load and Preprocess Data ---
    processed_data, user_mapping, item_mapping, n_users, n_items = load_and_preprocess_data(dummy_csv_path)

    # --- 2. Train Model ---
    output_model_path = 'ncf_model.h5'
    output_mappings_path = 'ncf_mappings.json'

    trained_model, training_history = train_model(
        processed_data, n_users, n_items,
        model_save_path=output_model_path,
        # mappings_save_path is handled separately below
        epochs=5 # Keep epochs low for example
    )

    # --- 3. Save Mappings ---
    save_mappings(user_mapping, item_mapping, output_mappings_path)

    print("Example training script finished.")
    print(f"Model saved to: {output_model_path}")
    print(f"Mappings saved to: {output_mappings_path}")

    # --- Example: Load mappings (demonstration) ---
    # loaded_user_map, loaded_item_map = load_mappings(output_mappings_path)
    # print(f"Successfully loaded user_map with {len(loaded_user_map)} users.")
    # print(f"Successfully loaded item_map with {len(loaded_item_map)} items.")
