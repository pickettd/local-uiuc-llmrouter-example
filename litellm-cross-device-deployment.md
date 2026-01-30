# LiteLLM Cross-Device Deployment Notes

Notes on deploying the `config.yaml` litellm proxy approach on a Raspberry Pi with Ollama.

## 1. Networking between LiteLLM and Ollama containers

The `config.yaml` uses the `ollama/` prefix, which makes LiteLLM call Ollama's API at `http://localhost:11434` by default. If Ollama runs in a separate container (or natively on the host) and LiteLLM runs in Docker, `localhost` inside the LiteLLM container won't reach Ollama. Options:

- Use `--network host` on the LiteLLM container (simplest)
- Add `api_base: http://host.docker.internal:11434` to each model's `litellm_params` in the config
- Run both in Docker on the same network and use the Ollama container name as the hostname

## 2. The `semantic-router` dependency

The `auto_router` feature depends on the `semantic-router` Python package (`pyproject.toml` explicitly lists it). The standard LiteLLM proxy Docker image may not include it. Options:

- Build a custom image that adds `pip install semantic-router`
- Check if newer litellm images bundle it (this may have changed since the feature was introduced)

## 3. The 30B model won't fit on a Pi

`qwen3-vl:30b-a3b-instruct-q4_K_M` at Q4 quantization still needs ~17-20GB of RAM. A Raspberry Pi 5 tops out at 8GB (or 16GB on the newest revision). That model won't run. The smaller models (`gemma3:1b`, `qwen3:0.6b`, `mxbai-embed-large`) should be fine on a Pi 4/5 with 8GB.

## 4. ARM64 image availability

Ollama has native ARM64 support and works well on Pi. The LiteLLM proxy Docker image (`ghcr.io/berriai/litellm`) publishes multi-arch images including ARM64, so that part should work.

## Summary

The approach is sound, but it's not a single-command deploy. You need to handle container networking, verify (or add) the `semantic-router` dependency in the litellm image, and swap out the 30B model for something that fits in Pi-sized RAM.
