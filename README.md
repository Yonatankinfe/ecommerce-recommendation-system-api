# E-commerce Recommendation System API

This project provides a "model-as-a-service" API for generating product recommendations based on user interaction data. It uses a Neural Collaborative Filtering (NCF) model.

## ✨ Features ✨

*   **Trainable Recommendation Model:** Upload your own dataset (CSV format) to train a custom recommendation model.
*   **NCF Architecture:** Utilizes a hybrid Neural Collaborative Filtering model to capture both linear and non-linear user-item interactions.
*   **RESTful API:**
    *   `POST /v1/train`: To upload data and initiate model training. Returns an API key upon completion.
    *   `POST /v1/recommendations`: To get product recommendations for a given user.
*   **API Key Authentication:** Secure access to the recommendation service.
*   **Containerized:** Easy to deploy using Docker.

## 🛠️ How it Works 🛠️

The system consists of three main components:

1.  **Data Ingestion & Training Service:**
    *   Accepts CSV data with `user_id`, `item_id`, and optional `interaction_score` and `timestamp`.
    *   Performs preprocessing:
        *   **Entity Mapping:** Maps `user_id`s and `item_id`s to integer indices.
        *   **Negative Sampling:** Generates negative samples for implicit feedback datasets.
    *   Trains an NCF model:
        *   **Embedding Layers:** Create dense vector representations for users and items.
        *   **GMF Path (Generalized Matrix Factorization):** Element-wise product of user and item embeddings to capture linear interactions.
        *   **MLP Path (Multi-Layer Perceptron):** Concatenated embeddings fed through dense layers to capture non-linear interactions.
        *   **Output Layer:** Combines GMF and MLP outputs with a Sigmoid activation for a probability score.
    *   Uses Binary Cross-Entropy loss and Adam optimizer.
    *   Serializes the trained model (.h5) and index mappings (JSON).
    *   (Future: Asynchronous training via a task queue).

2.  **Model Registry & Storage (Conceptual):**
    *   Trained models and mappings are stored (e.g., locally in a `models` directory, or conceptually in an S3-like object store for larger scale).
    *   Each model version is associated with an API key.

3.  **Inference API Service:**
    *   Built with FastAPI for high performance.
    *   Authenticates requests via `X-API-Key` header.
    *   `POST /v1/recommendations` endpoint:
        *   Loads the appropriate model and mappings based on the API key.
        *   Converts the input `user_id` to its integer index.
        *   Generates candidate items (all items not yet interacted with by the user).
        *   Predicts interaction scores for candidate items.
        *   Returns the top N recommended `item_id`s.
    *   (Future: In-memory caching for frequently accessed models and user recommendations).

## 🚀 Getting Started (High-Level) 🚀

1.  **Prepare your data:** Create a CSV file with columns `user_id`, `item_id`, and optionally `interaction_score`, `timestamp`.
2.  **Train a model:** Send a `POST` request to `/v1/train` with your CSV data. You'll receive an API key.
3.  **Get recommendations:** Send a `POST` request to `/v1/recommendations` with your `user_id` and the desired number of recommendations, including your API key in the `X-API-Key` header.

*(More detailed setup and API usage instructions will be added as development progresses, especially regarding running the Docker container.)*
