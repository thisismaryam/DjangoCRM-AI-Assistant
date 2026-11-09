import os
import json
import faiss
import numpy as np
from fastembed import TextEmbedding


def build_vector_index():
    # 1. Load the schema we extracted earlier
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    json_path = os.path.join(data_dir, 'schema_catalog.json')

    if not os.path.exists(json_path):
        print(
            f"Error: Could not find {json_path}. Run the extractor script first.")
        return

    with open(json_path, 'r') as f:
        schema_catalog = json.load(f)

    # 2. Initialize the embedding model (this downloads the model on first run)
    print("Loading embedding model...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # 3. Prepare the text and metadata
    model_names = []
    descriptions = []

    for model_name, data in schema_catalog.items():
        model_names.append(model_name)
        descriptions.append(data['description'])

    # 4. Generate vectors for all descriptions
    print(f"Generating vectors for {len(descriptions)} models...")
    # fastembed returns a generator, so we convert it to a list
    vectors = list(embedding_model.embed(descriptions))

    # Convert to a numpy array of float32 (required by FAISS)
    vectors_np = np.array(vectors).astype('float32')

    # 5. Build the FAISS Index
    # Get the dimension size from the embedding model (BGE-small is 384 dimensions)
    dimension = vectors_np.shape[1]

    # IndexFlatL2 is the simplest exact search index
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)

    # 6. Save the Index and the metadata
    index_path = os.path.join(data_dir, 'vector.index')
    faiss.write_index(index, index_path)

    # Save the mapping of index ID -> Model Name so we know what FAISS found later
    metadata_path = os.path.join(data_dir, 'index_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(model_names, f, indent=4)

    print(f"SUCCESS! Built FAISS index with {index.ntotal} vectors.")
    print(f"Saved to {index_path}")


if __name__ == "__main__":
    build_vector_index()
