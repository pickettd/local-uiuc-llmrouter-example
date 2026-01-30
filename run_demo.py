#!/usr/bin/env python3
"""End-to-end demo: train a KNN router and route queries to local Ollama models.

Prerequisites:
  1. pip install -r requirements.txt
  2. python generate_training_data.py  (generates data/ and knnrouter_config.yaml)
  3. Ollama running locally with models pulled (only needed for live inference)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIG_PATH = str(PROJECT_ROOT / "knnrouter_config.yaml")


def train_router():
    """Train the KNN router using LLMRouter's Python API."""
    from llmrouter.models.knnrouter import KNNRouter, KNNRouterTrainer

    print("Loading router config and data...")
    router = KNNRouter(yaml_path=CONFIG_PATH)

    print("Training KNN router...")
    trainer = KNNRouterTrainer(router=router, device="cpu")
    trainer.train()
    print("Router trained and saved.\n")
    return router


def route_query(router, query_text):
    """Route a single query and return the selected model name."""
    result = router.route_single({"query": query_text})
    return result["model_name"]


def call_ollama(model_name, query_text):
    """Call a local Ollama model via its OpenAI-compatible API."""
    import requests

    resp = requests.post(
        "http://localhost:11434/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": query_text}],
            "max_tokens": 256,
            "temperature": 0.7,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    # Ensure API_KEYS env var is set (Ollama needs empty string)
    os.environ.setdefault("API_KEYS", '{"Ollama": ""}')

    # Check that training data exists
    if not (PROJECT_ROOT / "data").exists():
        print("Error: data/ directory not found.")
        print("Run `python generate_training_data.py` first.")
        sys.exit(1)

    # --- Train ---
    print("=" * 60)
    print("STEP 1: Training KNN Router")
    print("=" * 60)
    router = train_router()

    # --- Route test queries ---
    print("=" * 60)
    print("STEP 2: Routing Test Queries")
    print("=" * 60)

    test_queries = [
        # Should route to gemma3:1b (coding)
        "Write a Python function to calculate fibonacci numbers",
        "Help me debug this JavaScript async function",
        "How do I implement a hash map from scratch",
        # Should route to qwen3:0.6b (simple)
        "What is the capital of Japan",
        "How many legs does a spider have",
        "What color is the sky",
        # Should route to qwen3-vl:30b (complex reasoning)
        "Prove that there are infinitely many prime numbers",
        "Design a distributed system architecture for real-time analytics",
        "Derive the quadratic formula from first principles",
    ]

    print(f"\n{'Query':<65} {'Routed To'}")
    print("-" * 110)
    for q in test_queries:
        model = route_query(router, q)
        print(f"{q:<65} {model}")

    # --- Optional: live inference with Ollama ---
    print()
    print("=" * 60)
    print("STEP 3: Live Inference (requires Ollama)")
    print("=" * 60)

    sample_query = "Write a Python function to check if a number is prime"
    model = route_query(router, sample_query)
    print(f"\nQuery:     {sample_query}")
    print(f"Routed to: {model}")

    try:
        print("\nCalling Ollama...")
        response = call_ollama(model, sample_query)
        print(f"\nResponse from {model}:")
        print("-" * 40)
        print(response)
    except Exception as e:
        print(f"\n(Ollama not available: {e})")
        print("Routing still works - Ollama is only needed for the actual LLM response.")


if __name__ == "__main__":
    main()
