# Cross-Device Deployment Notes

Transferring a trained KNN router (e.g., from an M3 Mac to a Jetson Nano or Raspberry Pi).

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

## Jetson Nano Constraints (Tested 2026-03-13)

The Jetson Nano Developer Kit (JetPack R32.7.6, Ubuntu 18.04, glibc 2.27) hits a **dependency wall** when trying to install the full inference stack natively:

1. **PyTorch 2.1+ requires `manylinux_2_28`** (glibc 2.28+). The Jetson's glibc 2.27 is too old — no compatible wheels exist.
2. **Pinning `torch==2.0.1`** gets past the wheel issue, but then:
   - `transformers>=5.0` requires PyTorch >= 2.4 and refuses to load models with 2.0.1
   - Pinning `transformers<4.40` (to get 4.39.3) gets both installed, but:
     - scikit-learn's bundled `libgomp` fails with `cannot allocate memory in static TLS block` (aarch64 glibc 2.27 bug)
     - `torch==2.0.1` was compiled against NumPy 1.x; current NumPy 2.x causes `_ARRAY_API not found` crashes
3. **Additional overrides needed:** `pyarrow<17` (cmake 3.25 required for source build, Jetson has 3.10.2), `hf-xet` excluded (build bug on aarch64 sdist).

**Bottom line:** Native installation on Jetson Nano with glibc 2.27 is not viable for the full UIUC LLMRouter stack.

### Docker as the solution (confirmed working 2026-03-17)

Docker is available on the Jetson Nano (Docker 20.10.21, NVIDIA runtime working, GPU devices visible in containers). The approach:

- Use a standard `python:3.11-slim` image (Debian Bookworm, glibc 2.36) which sidesteps all glibc issues
- Mount `~/llmrouter/` into the container
- KNN classification + Longformer embedding are CPU-only, so no CUDA needed
- The NVIDIA runtime is confirmed working (`/dev/nvhost-*` devices visible) if GPU access is ever needed

**Additional Docker-specific issues:**
- `torch==2.10.0` causes SIGILL (illegal instruction) on Cortex-A57 — newer PyTorch uses ARM ISA extensions the Tegra X1 doesn't support. **Fix:** `torch==2.6.0` (last version that runs on Cortex-A57).
- `torch-geometric` also causes SIGILL on Cortex-A57. Excluded since only needed for graph router, not KNN.
- `llmrouter.models.__init__` imports all router types (including graph router → torch-geometric → SIGILL). **Fix:** bypass the import chain with `route_query.py` that loads `.pkl` + Longformer directly via `from llmrouter.utils.embeddings import get_longformer_embedding`.

**Dockerfile:**
```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir "torch==2.6.0" "numpy<2" && \
    pip install --no-cache-dir "transformers<5" scikit-learn pyyaml pandas requests datasets litellm pillow && \
    pip install --no-cache-dir --no-deps llmrouter-lib
WORKDIR /app
```

**Run:**
```bash
docker build -t uiuc-router .
docker run --rm --network host -v /home/pickettd/llmrouter:/app uiuc-router python /app/route_query.py
```

**Results:** All 3 test queries routed correctly. Longformer embedding takes ~6s/query on Cortex-A57 CPU (49s cold start). KNN classification is ~20-30ms.

### GPU Acceleration on Jetson Nano (Tested 2026-03-17)

Two approaches tested to get GPU-accelerated Longformer embedding:

**`--runtime nvidia` with standard image: Won't work.** The `torch==2.6.0` aarch64 wheel from PyPI is CPU-only (`2.6.0+cpu`). PyTorch's CUDA wheels are only published for x86_64. The NVIDIA runtime correctly mounts GPU devices, but a CPU-only torch can't use them.

**`nvcr.io/nvidia/l4t-ml:r32.7.1-py3`: GPU works, stack too old.** This image has CUDA-enabled PyTorch 1.10 that sees the Tegra X1 GPU (`CUDA 10.2`). But it ships Python 3.6.9 + glibc 2.27 — the UIUC LLMRouter stack needs Python 3.10+ and modern transformers/scikit-learn, which can't run here.

**Possible future path (untested):** Build a custom image combining a newer base (Ubuntu 22.04+) with NVIDIA's Jetson-specific PyTorch wheel from `https://developer.download.nvidia.com/compute/redist/jp/`. Unknowns: wheel ABI compatibility with newer glibc/Python, and whether the Tegra X1's shared 4GB RAM leaves enough GPU memory for Longformer alongside llama.cpp.

**Current recommendation:** The CPU-only Docker approach (6s/query) is functional. GPU acceleration would be a nice-to-have but is blocked by the lack of CUDA-enabled aarch64 PyTorch on PyPI.

### Files deployed to Jetson

The following were copied to `~/llmrouter/` on the Jetson via scp:

- `pyproject.toml`, `run_demo.py`, `generate_training_data.py`, `ollama_models.json`
- `saved_models/knnrouter/knnrouter.pkl`
- `data/` (full directory — routing data, embeddings, LLM candidates)
- `knnrouter_config.yaml` (rewritten with `/home/pickettd/llmrouter/` absolute paths)

## Raspberry Pi Constraints

- **RAM**: Longformer needs ~570MB just for model weights, plus overhead. A Pi 4/5 with 4GB+ would handle it, but it would be tight on a 2GB model.
- **PyTorch on ARM**: Works on Pi 4/5 (aarch64), but you need ARM-compatible wheels. `pip install torch` handles this on recent Pi OS versions. Note: Pi OS is typically newer than the Jetson's Ubuntu 18.04, so glibc issues are less likely.
- **Inference latency**: Longformer embedding on CPU will be noticeably slower on a Pi's Cortex-A76 vs an M3.

## What to Transfer to the Target Device

- `saved_models/knnrouter/knnrouter.pkl` (small, just the KNN classifier)
- `knnrouter_config.yaml` (rewrite with target-device absolute paths)
- The full `data/` directory (routing data, embeddings, LLM candidates -- needed by `KNNRouter.__init__`)
- The codebase and `pyproject.toml`

The Longformer model auto-downloads from HuggingFace on first run, or you can copy the HuggingFace cache (`~/.cache/huggingface/`) to avoid downloading on the target device.

## Lighter Edge Deployment (Custom Path)

To avoid the full LLMRouter data loading overhead, you could bypass LLMRouter's `KNNRouter` class and write a minimal inference script that loads just the .pkl and a pre-downloaded Longformer directly. This would skip the `DataLoader` and YAML config requirements but would be outside LLMRouter's API.
