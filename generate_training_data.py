#!/usr/bin/env python3
"""Generate synthetic training data for LLMRouter KNN router with Ollama models.

This script creates all the data artifacts that LLMRouter needs to train a KNN router,
without requiring any actual LLM API calls. It:
  1. Defines ~45 queries across 3 categories (coding, simple, complex)
  2. Generates real Longformer embeddings for all queries
  3. Creates routing data with synthetic performance scores encoding our desired routing
  4. Writes all files to data/ and generates knnrouter_config.yaml with absolute paths
"""

import json
import os
import random
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.resolve()

# --- Model definitions ---

MODELS = {
    "coding": "gemma3:1b",
    "simple": "qwen3:0.6b",
    "complex": "qwen3-vl:30b-a3b-instruct-q4_K_M",
}

ALL_MODELS = list(MODELS.values())

# --- Synthetic queries by category ---

CODING_QUERIES = [
    "Write a Python function to reverse a string",
    "How do I implement a binary search in JavaScript",
    "Debug this code that has an off-by-one error in the loop",
    "Write a bash script to rename all files in a directory",
    "Explain how async await works in Python",
    "Help me write a REST API endpoint in Flask",
    "Write a SQL query to find duplicate rows in a table",
    "How do I sort a list of dictionaries by a specific key",
    "Write a recursive function to calculate factorial",
    "Create a Python class for a linked list with insert and delete methods",
    "How do I handle exceptions properly in Java",
    "Write a regular expression to validate email addresses",
    "Help me optimize this nested for loop that runs too slowly",
    "Write a unit test for this function using pytest",
    "How do I read and parse a CSV file in Python",
]

SIMPLE_QUERIES = [
    "What is the capital of France",
    "How many continents are there on Earth",
    "What year did World War 2 end",
    "Who wrote Romeo and Juliet",
    "What is the chemical symbol for water",
    "How many days are in a leap year",
    "What is the largest planet in our solar system",
    "Who painted the Mona Lisa",
    "What language do they speak in Brazil",
    "What is the boiling point of water in Celsius",
    "How many states are in the United States",
    "What is the currency of Japan",
    "Who invented the telephone",
    "What is the tallest mountain in the world",
    "What color do you get when you mix red and blue",
]

COMPLEX_QUERIES = [
    "Prove that the square root of 2 is irrational",
    "Design a fault-tolerant distributed consensus algorithm",
    "Evaluate the philosophical implications of hard determinism on moral responsibility",
    "Derive the Euler-Lagrange equation from the principle of least action",
    "Analyze the trade-offs between CAP theorem constraints for a global database",
    "Prove that there are infinitely many prime numbers using Euclids method",
    "Design a microservices architecture for a real-time trading platform",
    "Explain the relationship between Goedels incompleteness theorems and computability",
    "Derive the Black-Scholes equation for option pricing from first principles",
    "Analyze the computational complexity of the traveling salesman problem",
    "Prove the fundamental theorem of calculus rigorously",
    "Design a zero-knowledge proof system for identity verification",
    "Evaluate the epistemological differences between rationalism and empiricism",
    "Derive Maxwells equations from the electromagnetic field tensor",
    "Analyze the game-theoretic equilibria in multi-agent reinforcement learning",
]


def main():
    print(f"Project root: {PROJECT_ROOT}")

    # 1. Combine queries with their category labels
    queries = []
    for q in CODING_QUERIES:
        queries.append((q, "coding"))
    for q in SIMPLE_QUERIES:
        queries.append((q, "simple"))
    for q in COMPLEX_QUERIES:
        queries.append((q, "complex"))

    random.seed(42)
    random.shuffle(queries)

    # 2. Split 80/20 train/test
    split_idx = int(len(queries) * 0.8)
    train_queries = queries[:split_idx]
    test_queries = queries[split_idx:]
    print(f"Total queries: {len(queries)} (train: {len(train_queries)}, test: {len(test_queries)})")

    # 3. Generate Longformer embeddings for all queries
    from llmrouter.utils import get_longformer_embedding

    all_texts = [q for q, _ in queries]
    print(f"Generating Longformer embeddings for {len(all_texts)} queries...")
    embeddings = get_longformer_embedding(all_texts)  # shape: (N, 768)
    print(f"Embedding shape: {embeddings.shape}")

    # Convert to {int: tensor} dict format expected by LLMRouter
    embedding_dict = {i: embeddings[i] for i in range(len(all_texts))}

    # Build lookup: query text -> embedding_id
    query_to_emb_id = {q: i for i, (q, _) in enumerate(queries)}

    # 4. Create output directories
    data_dir = PROJECT_ROOT / "data"
    (data_dir / "query_data").mkdir(parents=True, exist_ok=True)
    (data_dir / "routing_data").mkdir(parents=True, exist_ok=True)
    (data_dir / "llm_candidates").mkdir(parents=True, exist_ok=True)
    os.makedirs(PROJECT_ROOT / "saved_models" / "knnrouter", exist_ok=True)

    # 5. Save query embeddings (.pt)
    pt_path = data_dir / "routing_data" / "query_embeddings_longformer.pt"
    torch.save(embedding_dict, pt_path)
    print(f"Saved embeddings to {pt_path}")

    # 6. Write query data JSONL files
    def write_query_jsonl(filepath, query_list):
        with open(filepath, "w") as f:
            for text, _cat in query_list:
                record = {
                    "task_name": "custom",
                    "query": text,
                    "ground_truth": "",
                    "metric": "custom",
                    "choices": None,
                    "task_id": None,
                }
                f.write(json.dumps(record) + "\n")

    qd_train = data_dir / "query_data" / "default_query_train.jsonl"
    qd_test = data_dir / "query_data" / "default_query_test.jsonl"
    write_query_jsonl(qd_train, train_queries)
    write_query_jsonl(qd_test, test_queries)
    print(f"Saved query data: {qd_train}, {qd_test}")

    # 7. Write routing data JSONL files
    # For each query, one row per candidate model. performance=1.0 for the intended best model.
    def write_routing_jsonl(filepath, query_list):
        with open(filepath, "w") as f:
            for text, cat in query_list:
                best_model = MODELS[cat]
                emb_id = query_to_emb_id[text]
                for model_name in ALL_MODELS:
                    perf = 1.0 if model_name == best_model else 0.0
                    record = {
                        "task_name": "custom",
                        "query": text,
                        "ground_truth": "",
                        "metric": "custom",
                        "choices": None,
                        "task_id": None,
                        "model_name": model_name,
                        "response": "",
                        "performance": perf,
                        "embedding_id": emb_id,
                        "token_num": 100,
                        "input_tokens": 50,
                        "output_tokens": 50,
                        "response_time": 1.0,
                        "success": True,
                    }
                    f.write(json.dumps(record) + "\n")

    rd_train = data_dir / "routing_data" / "default_routing_train_data.jsonl"
    rd_test = data_dir / "routing_data" / "default_routing_test_data.jsonl"
    write_routing_jsonl(rd_train, train_queries)
    write_routing_jsonl(rd_test, test_queries)
    print(f"Saved routing data: {rd_train}, {rd_test}")

    # 8. Copy LLM candidates JSON
    import shutil

    llm_src = PROJECT_ROOT / "ollama_models.json"
    llm_dst = data_dir / "llm_candidates" / "default_llm.json"
    shutil.copy(llm_src, llm_dst)
    print(f"Copied LLM candidates to {llm_dst}")

    # 9. Generate LLM feature embeddings
    llm_data = json.loads(llm_src.read_text())
    llm_embeddings = {}
    for name, info in llm_data.items():
        print(f"  Embedding LLM feature text for {name}...")
        emb = get_longformer_embedding(info["feature"])
        llm_embeddings[name] = {**info, "embedding": emb.tolist()}

    llm_emb_path = data_dir / "llm_candidates" / "default_llm_embeddings.json"
    with open(llm_emb_path, "w") as f:
        json.dump(llm_embeddings, f, indent=2)
    print(f"Saved LLM embeddings to {llm_emb_path}")

    # 10. Write knnrouter_config.yaml with absolute paths
    # (LLMRouter resolves relative paths against its own package root,
    #  so we use absolute paths to point to our project data.)
    config = {
        "data_path": {
            "query_data_train": str(qd_train),
            "query_data_test": str(qd_test),
            "query_embedding_data": str(pt_path),
            "routing_data_train": str(rd_train),
            "routing_data_test": str(rd_test),
            "llm_data": str(llm_dst),
            "llm_embedding_data": str(llm_emb_path),
        },
        "model_path": {
            "ini_model_path": "",
            "save_model_path": str(PROJECT_ROOT / "saved_models" / "knnrouter" / "knnrouter.pkl"),
            "load_model_path": str(PROJECT_ROOT / "saved_models" / "knnrouter" / "knnrouter.pkl"),
        },
        "metric": {
            "weights": {
                "performance": 1,
                "cost": 0,
                "llm_judge": 0,
            },
        },
        "hparam": {
            "n_neighbors": 3,
            "weights": "distance",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "metric": "cosine",
            "n_jobs": -1,
        },
    }

    config_path = PROJECT_ROOT / "knnrouter_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote config to {config_path}")

    print("\nDone! All training data generated.")
    print(f"  Data directory:  {data_dir}")
    print(f"  Config file:     {config_path}")
    print(f"\nNext step: python run_demo.py")


if __name__ == "__main__":
    main()
