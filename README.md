# SAMI Surgical Pruner for Dense GGUF Models
### Zero-Dequantization Byte-Level Layer Slicing for Transformers & Hybrid LLMs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-red.svg)](LICENSE)
[![Format: GGUF v3](https://img.shields.io/badge/Format-GGUF%20v3-orange.svg)](https://github.com/ggerganov/llama.cpp)
[![Hugging Face: MIThridate](https://img.shields.io/badge/Hugging%20Face-MIThridate-yellow.svg)](https://huggingface.co/MIThridate/Qwen3.5-9B-24L-GGUF)

> **SAMI Surgical Pruner** is an ultra-fast, byte-level utility for pruning redundant layers directly inside pre-quantized GGUF container files without dequantization, weight re-quantization, or GPU dependencies.

Developed by **Valentin Spiridon / TSV Consulting SRL** as part of the *Synapsa / Argos Ecosystem*.

---

## 📊 Verified Checkpoints (Qwen3.5-9B Suite)

Pre-compiled and verified GGUF weights produced by this tool are available on Hugging Face:  
🔗 [**huggingface.co/MIThridate/Qwen3.5-9B-24L-GGUF**](https://huggingface.co/MIThridate/Qwen3.5-9B-24L-GGUF)

| Model Checkpoint | Layers Dropped | Remaining Layers | Format | Size | Active VRAM | Tokens/sec | Recommended Workload |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **`Qwen3.5-9B-28L-Q4_K_M.gguf`** | 4 Layers (1 Period) | **28L** | Q4_K_M | **4.81 GB** | **~4.2 GB** | **18.0 tok/s** | 🏆 **PRODUCTION VERIFIED**: 100% stable on contract extraction (SLA 99.5%), legal compliance, structured reasoning, and zero looping. |
| **`Qwen3.5-9B-24L-SAMI-Q4_K_M.gguf`** | 8 Layers (2 Periods) | **24L** | SAMI-Q4 | **4.36 GB** | **~3.9 GB** | **21.5 tok/s** | ⚡ **Fast Edge Tier**: Ultra-compact 24L checkpoint for conversational assistants and factual retrieval on 4GB VRAM devices. |
| **`Qwen3.5-9B-24L-Q4_K_M.gguf`** | 8 Layers (2 Periods) | **24L** | Q4_K_M | **4.63 GB** | **~4.1 GB** | **20.8 tok/s** | 🧪 **24L Standard Baseline**: Sliced 8 central layers. Baseline experimental checkpoint. |
| **`Qwen3.5-9B-24L-Q3_K_M.gguf`** | 8 Layers (2 Periods) | **24L** | Q3_K_M | **3.79 GB** | **~3.4 GB** | **17.7 tok/s** | 📉 **Minimal Footprint**: Low memory environments and legacy hardware. |

---

## 🔬 Empirical Ablation Study: Layer-Dropping Horizons

During our empirical evaluation on the dense hybrid SSM-Transformer architecture **Qwen3.5-9B**, we tested multi-step reasoning, complex legal contract compliance, and code generation across pruning depths:

### 1. The 28-Layer Stability Horizon (Production Ready)
* **Periodic Cut:** 4 layers dropped (Period 3: alternating SSM/Attention block).
* **Empirical Integrity:** The residual stream completely absorbs the coordinate transition.
* **Empirical Results:**
  * **Legal Contract SLA Extraction:** Extracted exact **99.5% availability**, 1% per 0.1% penalty, and identified unilateral termination penalties without hallucination.
  * **Logical Reasoning:** Zero looping, stable softmax probability, clean execution exit.
  * **Throughput:** 18.0 tokens/second on commodity hardware (278 tokens/sec prompt processing).

### 2. The 24-Layer Boundary Condition (Edge Research Baseline)
* **Periodic Cut:** 8 layers dropped (Periods 3 & 4: 25% parameter reduction).
* **Empirical Observation:** While factual retrieval ("Capital of France = Paris") and conversational dialogue remain intact, dropping 8 unaligned layers without residual retraining marks the boundary where attention entropy rises on nested programming syntax.

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/MIThridate/sami-surgical-pruner.git
cd sami-surgical-pruner
pip install -e .
```

### 2. Inspect Model Structure
Inspect any GGUF file to see tensor counts, block structure, and attention periods:
```bash
python sami_gguf_layer_pruner.py --model Qwen3.5-9B-Q4_K_M.gguf --info-only
```

### 3. GGUF Direct Byte-Level Pruning (No Dequantization)
Prune redundant layers directly on pre-quantized GGUF files in under 20 seconds:
```bash
# Prune 1 period (4 layers) for production stability (28L):
python sami_gguf_layer_pruner.py \
    --model Qwen3.5-9B-Q4_K_M.gguf \
    --output Qwen3.5-9B-28L-Q4_K_M.gguf \
    --drop-periods 1

# Prune 2 periods (8 layers) for edge baseline (24L):
python sami_gguf_layer_pruner.py \
    --model Qwen3.5-9B-Q4_K_M.gguf \
    --output Qwen3.5-9B-24L-Q4_K_M.gguf \
    --drop-periods 2
```

Or drop specific layers by index:
```bash
python sami_gguf_layer_pruner.py \
    --model model.gguf \
    --output model_pruned.gguf \
    --drop-layers 12,13,14,15
```

### 4. Benchmark Model Locally
Verify VRAM consumption, generation speed, and multilingual coherence:
```bash
python benchmark_gpu.py Qwen3.5-9B-28L-Q4_K_M.gguf 99 2048
```

---

## ⚖️ License & Attribution

* **Software:** PolyForm Noncommercial License 1.0.0
* **Author:** Valentin Spiridon / TSV Consulting SRL (`contact@tsvconsulting.online`)
* **ORCID:** [0009-0000-2336-8527](https://orcid.org/0009-0000-2336-8527)
* **Citation:**  
  *Spiridon, V. (2026). SAMI Surgical Layer Pruner: Zero-Dequantization Block Dropper for Large Language Models. TSV Consulting Research.*
