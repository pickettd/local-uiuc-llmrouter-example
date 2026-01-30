# Cross-Device Deployment Notes

Transferring a trained KNN router (e.g., from an M3 Mac to a Raspberry Pi).

## The .pkl is Portable

The `saved_models/knnrouter/knnrouter.pkl` file is a scikit-learn `KNeighborsClassifier` saved via Python's `pickle`. It contains numpy arrays (training embeddings) using standard IEEE 754 floats, so they're architecture-independent. M3 Mac (ARM64) to Raspberry Pi (ARM64) works, and even x86 to ARM would work.

### Compatibility requirements for the .pkl

- Same major Python version on both machines (e.g., both 3.10.x or 3.11.x)
- Compatible scikit-learn versions (same minor version is safest; cross-minor usually works)
- Compatible numpy versions (generally fine across minor versions)

## Full Inference Stack

The .pkl alone isn't enough. When `route_single()` runs, it also:

1. **Loads Longformer** (`allenai/longformer-base-4096`, ~570MB) to embed the incoming query at inference time
2. **Requires the full YAML config + data files** because `KNNRouter.__init__` loads everything via `DataLoader` even if you only plan to call `route_single()`
3. **Needs PyTorch + Transformers** installed

## Raspberry Pi Constraints

- **RAM**: Longformer needs ~570MB just for model weights, plus overhead. A Pi 4/5 with 4GB+ would handle it, but it would be tight on a 2GB model.
- **PyTorch on ARM**: Works on Pi 4/5 (aarch64), but you need ARM-compatible wheels. `pip install torch` handles this on recent Pi OS versions.
- **Inference latency**: Longformer embedding on CPU will be noticeably slower on a Pi's Cortex-A76 vs an M3.

## What to Transfer to the Target Device

- `saved_models/knnrouter/knnrouter.pkl` (small, just the KNN classifier)
- `knnrouter_config.yaml` (rewrite with target-device absolute paths)
- The full `data/` directory (routing data, embeddings, LLM candidates -- needed by `KNNRouter.__init__`)
- The codebase and `pyproject.toml`

The Longformer model auto-downloads from HuggingFace on first run, or you can copy the HuggingFace cache (`~/.cache/huggingface/`) to avoid downloading on the target device.

## Lighter Edge Deployment (Custom Path)

To avoid the full LLMRouter data loading overhead, you could bypass LLMRouter's `KNNRouter` class and write a minimal inference script that loads just the .pkl and a pre-downloaded Longformer directly. This would skip the `DataLoader` and YAML config requirements but would be outside LLMRouter's API.
