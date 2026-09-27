#!/usr/bin/env python3
"""
SAMI FP16 Surgical Pruning Pipeline for Qwen3.5-9B.

Performs:
1. Removal of Vision Encoder (model.visual.*) — optional extraction to standalone vision bundle.
2. Surgical layer drop: 32 layers -> 24 layers (dropping center periods 3 & 4: layers 12-19).
3. Exact sequential reindexing of remaining layers (20-31 -> 12-23).
4. Config adaptation: text_config.num_hidden_layers = 24, layer_types adjusted to 24 blocks.
5. Memory-safe streaming I/O with automatic safetensors sharding (~4.5 GB / shard).
6. Copy of tokenizer, vocab, chat template, and metadata.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("sami_fp16_pruner")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAMI FP16 Pruner for Qwen3.5-9B")
    parser.add_argument(
        "--source_dir",
        type=str,
        default=r"C:\Users\TSVCo\.cache\huggingface\hub\models--Qwen--Qwen3.5-9B\snapshots\c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        help="Path to downloaded HF snapshot",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\TSVCo\Desktop\ARGOS_LIVE\compression\models\Qwen3.5-9B-pruned-24L-text",
        help="Path to output pruned model",
    )
    parser.add_argument(
        "--drop_layers",
        type=str,
        default="12,13,14,15,16,17,18,19",
        help="Comma-separated layer indices to drop (default: 12-19, 8 layers)",
    )
    parser.add_argument(
        "--extract_vision",
        action="store_true",
        default=True,
        help="Save extracted vision encoder as separate bundle",
    )
    parser.add_argument(
        "--max_shard_size_gb",
        type=float,
        default=4.5,
        help="Max size per safetensors shard in GB",
    )
    return parser


def parse_layers_to_drop(arg_str: str) -> list[int]:
    layers = [int(x.strip()) for x in arg_str.split(",") if x.strip()]
    return sorted(list(set(layers)))


def prune_model(args: argparse.Namespace):
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    drop_layers = parse_layers_to_drop(args.drop_layers)
    drop_set = set(drop_layers)

    LOG.info("=" * 70)
    LOG.info("  SAMI SURGICAL FP16 PRUNING PIPELINE")
    LOG.info(f"  Source:       {source_dir}")
    LOG.info(f"  Destination:  {output_dir}")
    LOG.info(f"  Drop Layers:  {drop_layers} ({len(drop_layers)} layers)")
    LOG.info("  Vision Out:   True (stripping model.visual.*)")
    LOG.info("=" * 70)

    # 1. Load index and config
    index_path = source_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        orig_index = json.load(f)

    config_path = source_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    orig_weight_map = orig_index["weight_map"]
    orig_total_size = orig_index.get("metadata", {}).get("total_size", 0)

    # 2. Plan layer remapping
    # Qwen3.5 has 32 layers
    total_orig_layers = 32
    if "text_config" in config and "num_hidden_layers" in config["text_config"]:
        total_orig_layers = config["text_config"]["num_hidden_layers"]

    kept_layers = [i for i in range(total_orig_layers) if i not in drop_set]
    layer_remap = {old_i: new_i for new_i, old_i in enumerate(kept_layers)}
    new_layer_count = len(kept_layers)

    LOG.info(f"Original layers: {total_orig_layers} -> New layers: {new_layer_count}")
    LOG.info(f"Kept layers: {kept_layers}")

    # 3. Categorize tensors
    tensors_to_keep: dict[str, tuple[str, str]] = {}  # new_name -> (shard_file, orig_name)
    vision_tensors: dict[str, str] = {}               # orig_name -> shard_file
    dropped_layer_tensors: list[str] = []

    layer_prefix = "model.language_model.layers."

    for orig_name, shard_file in orig_weight_map.items():
        if orig_name.startswith("model.visual."):
            vision_tensors[orig_name] = shard_file
        elif orig_name.startswith(layer_prefix):
            parts = orig_name.split(".")
            old_layer_idx = int(parts[3])
            if old_layer_idx in drop_set:
                dropped_layer_tensors.append(orig_name)
            else:
                new_layer_idx = layer_remap[old_layer_idx]
                new_name = f"{layer_prefix}{new_layer_idx}." + ".".join(parts[4:])
                tensors_to_keep[new_name] = (shard_file, orig_name)
        else:
            # Embeddings, lm_head, norms, mtp
            tensors_to_keep[orig_name] = (shard_file, orig_name)

    LOG.info(f"Total original tensors: {len(orig_weight_map)}")
    LOG.info(f"  - Dropped Vision tensors: {len(vision_tensors)}")
    LOG.info(f"  - Dropped Layer tensors:  {len(dropped_layer_tensors)}")
    LOG.info(f"  - Retained tensors:       {len(tensors_to_keep)}")

    # 4. Optional: Extract Vision Encoder
    if args.extract_vision and vision_tensors:
        LOG.info("\nExtracting Vision Encoder to companion bundle...")
        vision_dir = output_dir / "vision_encoder"
        vision_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and save vision tensors
        vision_state_dict = {}
        vision_shards = set(vision_tensors.values())
        for sfile in vision_shards:
            s_path = source_dir / sfile
            with safe_open(str(s_path), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith("model.visual."):
                        vision_state_dict[k] = f.get_tensor(k)
        
        vision_out_path = vision_dir / "vision_model.safetensors"
        save_safetensors(vision_state_dict, str(vision_out_path))
        vision_size_mb = os.path.getsize(vision_out_path) / (1024 * 1024)
        LOG.info(f"Vision Encoder extracted: {vision_out_path} ({vision_size_mb:.1f} MB)")
        del vision_state_dict
        gc.collect()

    # 5. Process and stream retained tensors into shards
    LOG.info("\nStreaming and packaging pruned text model shards...")
    max_shard_bytes = int(args.max_shard_size_gb * 1024 * 1024 * 1024)

    # Group tensors to fetch by their source shard to minimize file openings
    source_to_targets: dict[str, list[tuple[str, str]]] = {}
    for new_name, (src_shard, orig_name) in tensors_to_keep.items():
        source_to_targets.setdefault(src_shard, []).append((orig_name, new_name))

    current_shard_tensors: dict[str, torch.Tensor] = {}
    current_shard_bytes = 0
    saved_shards: list[str] = []
    new_weight_map: dict[str, str] = {}
    total_bytes_written = 0
    total_params_written = 0

    shard_idx = 1

    def flush_shard():
        nonlocal current_shard_tensors, current_shard_bytes, shard_idx, total_bytes_written
        if not current_shard_tensors:
            return
        shard_filename = f"model.safetensors-{shard_idx:05d}-of-PENDING.safetensors"
        shard_path = output_dir / shard_filename
        save_safetensors(current_shard_tensors, str(shard_path))
        actual_bytes = os.path.getsize(shard_path)
        total_bytes_written += actual_bytes
        LOG.info(f"  Saved temporary shard {shard_idx}: {actual_bytes / (1024*1024):.1f} MB")
        saved_shards.append(shard_filename)
        current_shard_tensors.clear()
        current_shard_bytes = 0
        shard_idx += 1
        gc.collect()

    t0 = time.time()
    tensor_count = 0
    total_to_process = len(tensors_to_keep)

    for src_shard, mapping in source_to_targets.items():
        src_path = source_dir / src_shard
        LOG.info(f"Reading from source shard: {src_shard} ({len(mapping)} tensors)...")
        with safe_open(str(src_path), framework="pt", device="cpu") as f:
            for orig_name, new_name in mapping:
                tensor = f.get_tensor(orig_name)
                t_bytes = tensor.numel() * tensor.element_size()
                total_params_written += tensor.numel()

                if current_shard_bytes + t_bytes > max_shard_bytes and current_shard_tensors:
                    flush_shard()

                current_shard_tensors[new_name] = tensor
                current_shard_bytes += t_bytes
                tensor_count += 1

                # Record shard mapping placeholder
                new_weight_map[new_name] = shard_idx

    flush_shard()

    total_shards = len(saved_shards)
    LOG.info(f"Finalizing {total_shards} shards with exact names...")

    # Rename shards with actual count: -XXXXX-of-YYYYY.safetensors
    final_shard_names = {}
    for i in range(1, total_shards + 1):
        old_name = f"model.safetensors-{i:05d}-of-PENDING.safetensors"
        new_name = f"model.safetensors-{i:05d}-of-{total_shards:05d}.safetensors"
        os.rename(output_dir / old_name, output_dir / new_name)
        final_shard_names[i] = new_name

    # Build final weight map
    final_weight_map = {k: final_shard_names[s_id] for k, s_id in new_weight_map.items()}

    # 6. Save new index
    new_index = {
        "metadata": {
            "total_size": total_bytes_written,
        },
        "weight_map": final_weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2)

    # 7. Update and save config.json
    new_config = json.loads(json.dumps(config))  # deep copy
    if "text_config" in new_config:
        new_config["text_config"]["num_hidden_layers"] = new_layer_count
        if "layer_types" in new_config["text_config"]:
            # Periodic 4-layer pattern repeated 6 times (24 layers)
            new_config["text_config"]["layer_types"] = [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ] * (new_layer_count // 4)

    # Strip vision references from main config
    for v_key in [
        "vision_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ]:
        new_config.pop(v_key, None)

    # Update architectures if applicable
    new_config["architectures"] = ["Qwen3_5ForCausalLM"]

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=2)

    # 8. Copy tokenizer and helper files
    copy_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    ]
    for cfile in copy_files:
        src_f = source_dir / cfile
        if src_f.exists():
            shutil.copy2(src_f, output_dir / cfile)
            LOG.info(f"Copied {cfile}")

    elapsed = time.time() - t0
    orig_gb = orig_total_size / (1024**3)
    new_gb = total_bytes_written / (1024**3)
    reduction = (1.0 - new_gb / max(orig_gb, 0.001)) * 100

    LOG.info("\n" + "=" * 70)
    LOG.info("  PRUNING COMPLETE SUCCESS!")
    LOG.info(f"  Original FP16 Model Size: {orig_gb:.2f} GB")
    LOG.info(f"  Pruned FP16 Model Size:   {new_gb:.2f} GB ({reduction:.1f}% reduction)")
    LOG.info(f"  Total Active Parameters:  {total_params_written:,} ({total_params_written/1e9:.2f}B)")
    LOG.info(f"  Elapsed Time:             {elapsed:.1f}s")
    LOG.info(f"  Output Directory:         {output_dir}")
    LOG.info("=" * 70)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    prune_model(args)
