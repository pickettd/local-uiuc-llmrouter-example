#!/usr/bin/env python3
"""Minimal KNN routing script for Jetson — bypasses llmrouter import chain."""

import pickle
import sys
import time
import json
import numpy as np
from llmrouter.utils.embeddings import get_longformer_embedding


def load_router(pkl_path):
    """Load the trained KNN classifier from pickle."""
    with open(pkl_path, "rb") as f:
        knn = pickle.load(f)
    return knn


def route(knn, query, label_map):
    """Embed query with Longformer, classify with KNN, return model name."""
    t0 = time.time()
    embedding = get_longformer_embedding(query).numpy().reshape(1, -1)
    t_embed = time.time() - t0

    t0 = time.time()
    label = knn.predict(embedding)[0]
    t_knn = time.time() - t0

    # KNN labels may be ints (mapped via label_map) or model name strings directly
    if isinstance(label, (int, np.integer)):
        model_name = label_map.get(label, f"unknown-label-{label}")
    else:
        model_name = str(label)
    return model_name, t_embed, t_knn


def main():
    pkl_path = "/app/saved_models/knnrouter/knnrouter.pkl"
    llm_data_path = "/app/data/llm_candidates/default_llm.json"

    # Load model name mapping: label index -> model name
    with open(llm_data_path) as f:
        llm_data = json.load(f)
    label_map = {i: name for i, name in enumerate(llm_data.keys())}
    print(f"Label map: {label_map}")

    # Load KNN
    print("Loading KNN classifier...")
    knn = load_router(pkl_path)
    print(f"KNN loaded: {knn.n_neighbors} neighbors, {knn.metric} metric")

    # Route test queries
    queries = sys.argv[1:] if len(sys.argv) > 1 else [
        "What is 2+2?",
        "Write a Python function to sort a list",
        "Prove that there are infinitely many prime numbers",
    ]

    for q in queries:
        model, t_embed, t_knn = route(knn, q, label_map)
        print(f"Query: {q}")
        print(f"  Routed to: {model}  (embed: {t_embed:.3f}s, knn: {t_knn:.4f}s)")


if __name__ == "__main__":
    main()
