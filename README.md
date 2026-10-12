# sami-surgical-pruner

CLI tool for excising transformer layers directly from GGUF binaries without dequantization or GPU dependencies.

---

## Overview

Modifying layer depth in quantized models typically requires loading full-precision checkpoints into PyTorch, dropping modules, saving new weights, and repeating the quantization process.

`sami_gguf_layer_pruner.py` operates directly on pre-quantized GGUF files (`Q4_K`, `Q5_K`, `Q8_0`, etc.):
* Reads the GGUF binary container and identifies tensor name schemas.
* Omits selected layer tensors during sequential write.
* Renumbers subsequent layer tensors to maintain sequential index continuity.
* Rewrites metadata header keys (`*.block_count`) to reflect the new architecture.
* Runs purely on CPU using system RAM, processing multi-gigabyte models in seconds.

---

## Experimental Status & Community Feedback

This tool is an experimental proof-of-concept developed for hardware-constrained inference research.

* **Work in progress:** Structural layer excision without subsequent retraining is inherently empirical. Behavior, perplexity drift, and degradation thresholds will vary across different base architectures.
* **Feedback:** Observations, benchmark logs, and issue reports across different model families are welcome. If you encounter edge-case failures, token-looping anomalies, or test new architectures, please open an **Issue** or submit a **Pull Request**.

---

## Requirements & Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/MIThriade/sami-surgical-pruner.git
cd sami-surgical-pruner
pip install -r requirements.txt
```

---

## Usage

### 1. Inspect Model Architecture

Display layer counts, tensor names, and metadata without modifying the file:

```bash
python sami_gguf_layer_pruner.py --model Qwen3.5-9B-Q4_K_M.gguf --info-only
```

### 2. Drop Layers by Period (Hybrid Architectures)

For models with periodic layer repeating patterns (e.g., Qwen3.5 alternating SSM/Attention blocks in 4-layer cycles):

```bash
# Drop 1 period (4 layers) -> 28 layers remaining
python sami_gguf_layer_pruner.py \
    --model Qwen3.5-9B-Q4_K_M.gguf \
    --output Qwen3.5-9B-28L-Q4_K_M.gguf \
    --drop-periods 1

# Drop 2 periods (8 layers) -> 24 layers remaining
python sami_gguf_layer_pruner.py \
    --model Qwen3.5-9B-Q4_K_M.gguf \
    --output Qwen3.5-9B-24L-Q4_K_M.gguf \
    --drop-periods 2
```

### 3. Drop Explicit Layer Indices

Specify exact comma-separated layer indices to excise:

```bash
python sami_gguf_layer_pruner.py \
    --model input.gguf \
    --output output_pruned.gguf \
    --drop-layers 12,13,14,15
```

---

## Engineering Considerations

* **No Retraining:** This tool performs structural ablation. Removing layers introduces perplexity degradation.
* **Architecture Constraints:** In hybrid architectures (such as Mamba-2 / GQA), dropping arbitrary single layers can disrupt alternating state-space cycles. It is recommended to drop full architectural periods.
* **Verification:** Always measure downstream perplexity and token coherence on your target use case following pruning.

---

## Checkpoints

Pre-pruned reference checkpoints derived from `Qwen/Qwen3.5-9B` using this tool:

* **Model repository:** [https://huggingface.co/MIThridate/Qwen3.5-9B-24L-GGUF](https://huggingface.co/MIThridate/Qwen3.5-9B-24L-GGUF)

---

## Author & Citation

* **Author:** Valentin Spiridon (TSV Consulting SRL, Timișoara, Romania)

```bibtex
@software{spiridon2026sami_pruner,
  author       = {Spiridon, Valentin},
  title        = {SAMI Surgical Pruner: Zero-Dequantization Layer Slicing for Quantized GGUF Models},
  year         = {2026},
  url          = {https://github.com/MIThriade/sami-surgical-pruner}
}
```

---

## License

Apache-2.0
