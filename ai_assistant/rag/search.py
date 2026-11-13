import os
import json
import faiss
import numpy as np
from fastembed import TextEmbedding

_embedding_model = None
_index = None
_metadata = None
_catalog = None

# Expansion tuning (Phase 2.2)
MAX_TOTAL_SCHEMAS = 8            # hard cap: matched + expanded
MAX_TRAVERSAL_DEPTH = 1          # 1-hop only

# Reference/lookup models: still matchable, never auto-expanded.
LOOKUP_MODELS = {
    "City",
    "Country",
    "ClientType",
    "Industry",
    "Stage",
    "LeadSource",
    "ClosingReason",
    "Tag",
    "Currency",
    "ProductCategory",
    "ContinentalRegion",
    "SubRegion",
    "IntermediaryRegion",
    "CategoryPriceRule",
    "DepartmentPriceRule",
    "ProductPriceTier",
}


def _load_resources():
    global _embedding_model, _index, _metadata, _catalog

    if _embedding_model is None:
        print("Loading search model into memory...")
        _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

        index_path = os.path.join(data_dir, 'vector.index')
        _index = faiss.read_index(index_path)

        metadata_path = os.path.join(data_dir, 'index_metadata.json')
        with open(metadata_path, 'r') as f:
            _metadata = json.load(f)

        catalog_path = os.path.join(data_dir, 'schema_catalog.json')
        with open(catalog_path, 'r', encoding='utf-8') as f:
            _catalog = json.load(f)


def search_schema(query: str, top_k: int = 3):
    """Return the top_k most relevant models (unchanged)."""
    _load_resources()

    query_vector = list(_embedding_model.embed([query]))[0]
    query_np = np.array([query_vector]).astype('float32')

    distances, indices = _index.search(query_np, top_k)

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        if idx != -1:
            results.append({
                "model": _metadata[idx],
                "distance": float(distances[0][i]),
            })
    return results


def _connected_models(model_name: str) -> list[str]:
    """1-hop neighbours, both directions, proxy-resolved, lookups excluded."""
    entry = _catalog.get(model_name)
    if not entry:
        return []

    if entry.get("proxy"):
        concrete = entry.get("concrete_model")
        return [concrete] if concrete and concrete in _catalog else []

    neighbours = set()

    for rel in entry.get("relationships", []):
        target = rel.get("related_model")
        if target and target in _catalog:
            neighbours.add(target)

    for rel in entry.get("reverse_relationships", []):
        source = rel.get("related_model")
        if source and source in _catalog:
            neighbours.add(source)

    resolved = set()
    for name in neighbours:
        n_entry = _catalog.get(name, {})
        if n_entry.get("proxy") and n_entry.get("concrete_model") in _catalog:
            resolved.add(n_entry["concrete_model"])
        else:
            resolved.add(name)

    resolved.discard(model_name)
    resolved -= LOOKUP_MODELS
    return sorted(resolved)


def search_schema_with_relationships(query: str, top_k: int = 3):
    """
    Returns matched models AND their 1-hop business neighbours, with full
    schema for each. Total schemas capped at MAX_TOTAL_SCHEMAS.
    """
    _load_resources()

    matches = search_schema(query, top_k=top_k)
    if not matches:
        return {"matched_models": [], "expanded_models": [], "schemas": {}}

    matched_names = [m["model"] for m in matches]

    collected = list(dict.fromkeys(matched_names))
    frontier = list(collected)
    depth = 0

    while frontier and depth < MAX_TRAVERSAL_DEPTH:
        if len(collected) >= MAX_TOTAL_SCHEMAS:
            break
        next_frontier = []
        for name in frontier:
            for neighbour in _connected_models(name):
                if neighbour in collected:
                    continue
                if len(collected) >= MAX_TOTAL_SCHEMAS:
                    break
                collected.append(neighbour)
                next_frontier.append(neighbour)
        frontier = next_frontier
        depth += 1

    # Hard cap on returned schemas (matched models may already exceed cap
    # if top_k > MAX_TOTAL_SCHEMAS — truncate in that case, but keep them
    # in matched_models for transparency).
    schema_names = collected[:MAX_TOTAL_SCHEMAS]
    expanded_only = [n for n in schema_names if n not in matched_names]

    schemas = {name: _catalog[name]
               for name in schema_names if name in _catalog}

    return {
        "matched_models": matched_names,
        "expanded_models": expanded_only,
        "schemas": schemas,
    }


if __name__ == "__main__":
    print("\n--- Testing RAG Search (raw) ---")

    for q in [
        "Where do I store sales deals?",
        "How do I track a person's phone number?",
        "Where is the company address saved?",
    ]:
        print(f"\nQuery: '{q}'")
        for match in search_schema(q, top_k=2):
            print(
                f"  -> Match: {match['model']} (Distance: {match['distance']:.4f})")

    print("\n--- Testing relationship expansion ---")

    for q in [
        "Show me payments for a company",
        "How are payments connected to companies?",
        "Which shipments belong to this deal?",
        "Which contacts belong to a company?",
    ]:
        print(f"\nQuery: '{q}'")
        result = search_schema_with_relationships(q, top_k=2)
        print(f"  matched : {result['matched_models']}")
        print(f"  expanded: {result['expanded_models']}")
        print(f"  schemas : {list(result['schemas'].keys())} "
              f"(count={len(result['schemas'])})")
