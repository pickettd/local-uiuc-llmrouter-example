# Local LLMRouter Example with Ollama

Minimal working example of [UIUC LLMRouter](https://github.com/ulab-uiuc/LLMRouter) routing queries to local Ollama models using a KNN router.

## What This Does

Routes queries to 3 local Ollama models based on query complexity:

| Query Type | Model | Size |
|---|---|---|
| Coding tasks | `gemma3:1b` | 1B params |
| Simple/general questions | `qwen3:0.6b` | 0.6B params |
| Complex reasoning | `qwen3-vl:30b-a3b-instruct-q4_K_M` | 30B params |

This replicates the routing intent from `litellm-config.yaml` (rule-based semantic routing) using LLMRouter's ML-based KNN approach instead.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Ollama running locally (only needed for live inference, not for routing)

```bash
ollama pull gemma3:1b
ollama pull qwen3:0.6b
ollama pull qwen3-vl:30b-a3b-instruct-q4_K_M
```

## Quickstart

```bash
# Install dependencies
uv sync

# Generate synthetic training data + Longformer embeddings
uv run python generate_training_data.py

# Train KNN router + demo routing + call Ollama
uv run python run_demo.py
```

## How It Works

1. **`generate_training_data.py`** creates synthetic training data:
   - 45 queries across 3 categories (coding, simple, complex)
   - Real [Longformer](https://huggingface.co/allenai/longformer-base-4096) embeddings (768-dim) for semantic similarity
   - Synthetic performance scores encoding which model is "best" for each query type
   - Writes all files to `data/` and generates `knnrouter_config.yaml`

2. **`run_demo.py`** trains and demonstrates the router:
   - Trains a KNN classifier on (query_embedding, best_model) pairs
   - Routes new queries by finding the k=3 nearest training queries (cosine distance)
   - Optionally calls the selected Ollama model for a live response

## CLI Alternative

After generating data, you can also use LLMRouter's CLI directly:

```bash
export API_KEYS='{"Ollama": ""}'

# Train
uv run llmrouter train --router knnrouter --config knnrouter_config.yaml

# Route a single query
uv run llmrouter infer --router knnrouter --config knnrouter_config.yaml \
  --query "Write a sort function" --route-only

# Interactive chat UI
uv run llmrouter chat --router knnrouter --config knnrouter_config.yaml --port 8001
```

## Project Structure

```
.
├── README.md
├── pyproject.toml              # Project config + dependencies (managed by uv)
├── uv.lock                     # Locked dependency versions
├── ollama_models.json           # LLM candidate definitions (3 Ollama models)
├── litellm-config.yaml          # Reference: original LiteLLM semantic routing config
├── knnrouter_config.yaml        # Generated: KNN router config with absolute paths
├── generate_training_data.py    # Generates all training data + config
├── run_demo.py                  # Trains router + demos routing + Ollama inference
├── data/                        # Generated training data (gitignored)
│   ├── query_data/
│   ├── routing_data/
│   └── llm_candidates/
└── saved_models/                # Trained router model (gitignored)
```

## Background: LiteLLM vs LLMRouter

| | LiteLLM Semantic Routing | UIUC LLMRouter |
|---|---|---|
| Approach | Rule-based utterance similarity | ML-trained (KNN, SVM, MLP, etc.) |
| Embeddings | Custom (mxbai-embed-large via Ollama) | Longformer (allenai/longformer-base-4096) |
| Training | None (configure utterances directly) | Requires training data |
| Routing strategies | 1 (semantic threshold) | 16+ (KNN, SVM, GNN, etc.) |
