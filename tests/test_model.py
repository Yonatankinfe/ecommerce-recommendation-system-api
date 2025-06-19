import pytest
import tensorflow as tf
from src.model import create_ncf_model # Adjusted import path

# Basic NCF Model Tests
def test_create_ncf_model_output_shape():
    num_users, num_items, embedding_dim = 100, 50, 32
    model = create_ncf_model(num_users, num_items, embedding_dim=embedding_dim)

    # Check input shapes (batch_size, 1)
    user_input_shape = model.input_shape[0] # User input
    item_input_shape = model.input_shape[1] # Item input
    assert user_input_shape == (None, 1)
    assert item_input_shape == (None, 1)

    # Check output shape (batch_size, 1)
    output_shape = model.output_shape
    assert output_shape == (None, 1)

def test_create_ncf_model_compiles():
    num_users, num_items = 100, 50
    model = create_ncf_model(num_users, num_items)
    try:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
        compiled = True
    except Exception as e:
        print(f"Model compilation failed: {e}")
        compiled = False
    assert compiled

def test_ncf_model_layer_names():
    num_users, num_items = 100, 50
    model = create_ncf_model(num_users, num_items, mlp_layers=[64, 32])

    expected_gmf_user_embedding_name = 'gmf_user_embedding'
    expected_mlp_user_embedding_name = 'mlp_user_embedding'
    expected_gmf_multiply_name = 'gmf_multiply'
    expected_mlp_concatenate_name = 'mlp_concatenate'
    expected_neumf_concatenate_name = 'neumf_concatenate'
    expected_output_layer_name = 'output_layer'

    layer_names = [layer.name for layer in model.layers]

    assert expected_gmf_user_embedding_name in layer_names
    assert expected_mlp_user_embedding_name in layer_names
    assert expected_gmf_multiply_name in layer_names
    assert expected_mlp_concatenate_name in layer_names
    assert expected_neumf_concatenate_name in layer_names
    assert expected_output_layer_name in layer_names
    assert f'mlp_dense_layer_0' in layer_names # Check for first MLP dense layer
    assert f'mlp_dense_layer_1' in layer_names # Check for second MLP dense layer

def test_ncf_model_predict_smoke():
    num_users, num_items = 10, 5
    model = create_ncf_model(num_users, num_items)
    model.compile(optimizer='adam', loss='binary_crossentropy') # Compile before predict

    # Dummy input data: 2 samples
    user_data = tf.constant([[0], [1]], dtype=tf.int32) # User indices
    item_data = tf.constant([[0], [1]], dtype=tf.int32) # Item indices

    try:
        predictions = model.predict([user_data, item_data])
        assert predictions.shape == (2, 1) # Batch size 2, output dim 1
        assert ((predictions >= 0) & (predictions <= 1)).all() # Sigmoid output
    except Exception as e:
        pytest.fail(f"Model prediction failed: {e}")
