import os
import json
import faiss
import numpy as np
from fastembed import TextEmbedding

# We load these globally so they stay in memory
_embedding_model = None
_index = None
_metadata = None


def _load_resources():
    global _embedding_model, _index, _metadata

    if _embedding_model is None:
        print("Loading search model into memory...")
        _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

        # Load FAISS index
        index_path = os.path.join(data_dir, 'vector.index')
        _index = faiss.read_index(index_path)

        # Load Metadata (the list of model names)
        metadata_path = os.path.join(data_dir, 'index_metadata.json')
        with open(metadata_path, 'r') as f:
            _metadata = json.load(f)


def search_schema(query: str, top_k: int = 3):
    """
    Takes a natural language query and returns the top_k most relevant models.
    """
    _load_resources()

    # 1. Embed the user's query
    query_vector = list(_embedding_model.embed([query]))[0]
    query_np = np.array([query_vector]).astype('float32')

    # 2. Search the FAISS index
    # distances: how close the match is. indices: the ID of the matched vector
    distances, indices = _index.search(query_np, top_k)

    # 3. Map the indices back to the model names
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        # If the index is valid
        if idx != -1:
            model_name = _metadata[idx]
            distance = distances[0][i]
            results.append({
                "model": model_name,
                "distance": float(distance)
            })

    return results


# This block runs if we test the file directly
if __name__ == "__main__":
    print("\n--- Testing RAG Search ---")

    test_queries = [
        "Where do I store sales deals?",
        "How do I track a person's phone number?",
        "Where is the company address saved?"
    ]

    for q in test_queries:
        print(f"\nQuery: '{q}'")
        matches = search_schema(q, top_k=2)
        for match in matches:
            print(
                f"  -> Match: {match['model']} (Distance: {match['distance']:.4f})")
