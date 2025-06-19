import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Multiply, Concatenate, Dense
from tensorflow.keras.models import Model

def create_ncf_model(num_users, num_items, embedding_dim=32, mlp_layers=[64, 32, 16], reg_mlp=0, reg_gmf=0):
    """
    Creates a Neural Collaborative Filtering (NCF) model.

    Args:
        num_users (int): Total number of unique users.
        num_items (int): Total number of unique items.
        embedding_dim (int): Dimensionality of the embedding layers.
        mlp_layers (list of int): List of hidden layer sizes for the MLP path.
        reg_mlp (float): L2 regularization factor for MLP layers.
        reg_gmf (float): L2 regularization factor for GMF embedding layers.

    Returns:
        tensorflow.keras.models.Model: The compiled NCF model.
    """

    # Input layers
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # --- GMF (Generalized Matrix Factorization) Path ---
    # Embedding layer for GMF
    gmf_user_embedding = Embedding(input_dim=num_users,
                                   output_dim=embedding_dim,
                                   embeddings_initializer='he_normal',
                                   embeddings_regularizer=tf.keras.regularizers.l2(reg_gmf),
                                   input_length=1,
                                   name='gmf_user_embedding')
    gmf_item_embedding = Embedding(input_dim=num_items,
                                   output_dim=embedding_dim,
                                   embeddings_initializer='he_normal',
                                   embeddings_regularizer=tf.keras.regularizers.l2(reg_gmf),
                                   input_length=1,
                                   name='gmf_item_embedding')

    # Flatten embedding outputs
    gmf_user_latent = Flatten()(gmf_user_embedding(user_input))
    gmf_item_latent = Flatten()(gmf_item_embedding(item_input))

    # Element-wise product
    gmf_vector = Multiply(name='gmf_multiply')([gmf_user_latent, gmf_item_latent])

    # --- MLP (Multi-Layer Perceptron) Path ---
    # Embedding layer for MLP
    mlp_user_embedding = Embedding(input_dim=num_users,
                                   output_dim=embedding_dim,
                                   embeddings_initializer='he_normal',
                                   embeddings_regularizer=tf.keras.regularizers.l2(reg_mlp),
                                   input_length=1,
                                   name='mlp_user_embedding')
    mlp_item_embedding = Embedding(input_dim=num_items,
                                   output_dim=embedding_dim,
                                   embeddings_initializer='he_normal',
                                   embeddings_regularizer=tf.keras.regularizers.l2(reg_mlp),
                                   input_length=1,
                                   name='mlp_item_embedding')

    # Flatten embedding outputs
    mlp_user_latent = Flatten()(mlp_user_embedding(user_input))
    mlp_item_latent = Flatten()(mlp_item_embedding(item_input))

    # Concatenate user and item embeddings
    mlp_vector = Concatenate(name='mlp_concatenate')([mlp_user_latent, mlp_item_latent])

    # MLP layers
    for i, units in enumerate(mlp_layers):
        mlp_vector = Dense(units,
                           activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(reg_mlp),
                           name=f'mlp_dense_layer_{i}')(mlp_vector)

    # --- Concatenate GMF and MLP paths ---
    # NeuMF (Neural Matrix Factorization) layer
    neumf_vector = Concatenate(name='neumf_concatenate')([gmf_vector, mlp_vector])

    # Output layer
    output = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='output_layer')(neumf_vector)

    # Create and compile model
    model = Model(inputs=[user_input, item_input], outputs=output, name='NCF_Model')

    # It's generally better to compile the model in the training script
    # where the optimizer and loss function are more contextually relevant.
    # However, providing a default compile step here for completeness.
    # model.compile(optimizer=tf.keras.optimizers.Adam(),
    #               loss='binary_crossentropy',
    #               metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    return model

if __name__ == '__main__':
    # Example Usage:
    print("Creating a sample NCF model...")
    num_users_example = 1000
    num_items_example = 500

    ncf_model = create_ncf_model(num_users_example, num_items_example)
    ncf_model.summary()
    print("NCF model created successfully.")

    # You can save a plot of the model architecture
    # tf.keras.utils.plot_model(ncf_model, to_file='ncf_model_architecture.png', show_shapes=True)
    # print("Model architecture plot saved to ncf_model_architecture.png")
