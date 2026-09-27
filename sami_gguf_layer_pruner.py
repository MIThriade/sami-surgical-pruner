#!/usr/bin/env python3
"""
SAMI GGUF Layer Pruner - High-Performance Block Dropping for Quantized GGUF Models.

Performs zero-dequantization, byte-exact pruning of transformer blocks directly on
quantized GGUF files (Q4_K_M, NVFP4, Q5_K, F16, etc.).

Preserves:
- Native quantization format & weight tables byte-for-byte
- Tokenizer vocab, merges, and chat templates
- RoPE scaling, SSM configs, and model metadata
- Sequentially reindexes remaining blocks to avoid KV-cache corruption
"""

import sys
import os
import struct
import argparse
import time
from pathlib import Path
import numpy as np

# GGUF Constants
GGUF_MAGIC = 0x46554747  # 'GGUF'
GGUF_DEFAULT_ALIGNMENT = 32

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML Quantization Sizes (block_size, type_size_bytes)
GGML_QUANT_SIZES = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    9: (32, 36),    # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    15: (256, 292), # Q8_K
    16: (16, 2),    # IQ2_XXS
    17: (16, 2),    # IQ2_XS
    18: (16, 2),    # IQ3_XXS
    19: (32, 4),    # IQ1_S
    20: (32, 4),    # IQ4_NL
    21: (32, 4),    # IQ3_S
    22: (32, 4),    # IQ2_S
    23: (32, 4),    # IQ4_XS
    24: (1, 2),     # I8
    25: (1, 2),     # I16
    26: (1, 4),     # I32
    27: (1, 8),     # I64
    28: (1, 8),     # F64
    29: (32, 4),    # IQ1_M
    30: (1, 2),     # BF16
    39: (32, 17),   # MXFP4
    40: (16, 9),    # NVFP4
}

def pad(val: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    return ((val + alignment - 1) // alignment) * alignment

def read_str_raw(f) -> tuple[str, bytes]:
    len_bytes = f.read(8)
    length = struct.unpack('<Q', len_bytes)[0]
    str_bytes = f.read(length)
    return str_bytes.decode('utf-8', errors='replace'), len_bytes + str_bytes

def write_str(s: str) -> bytes:
    encoded = s.encode('utf-8')
    return struct.pack('<Q', len(encoded)) + encoded

def read_kv_pair_raw(f) -> tuple[str, int, bytes, any]:
    key, key_raw = read_str_raw(f)
    vtype_bytes = f.read(4)
    vtype = struct.unpack('<I', vtype_bytes)[0]
    
    val_raw, val_parsed = read_val_raw(f, vtype)
    return key, vtype, key_raw + vtype_bytes + val_raw, val_parsed

def read_val_raw(f, vtype: int) -> tuple[bytes, any]:
    if vtype == GGUF_TYPE_UINT8:
        b = f.read(1); return b, struct.unpack('<B', b)[0]
    elif vtype == GGUF_TYPE_INT8:
        b = f.read(1); return b, struct.unpack('<b', b)[0]
    elif vtype == GGUF_TYPE_UINT16:
        b = f.read(2); return b, struct.unpack('<H', b)[0]
    elif vtype == GGUF_TYPE_INT16:
        b = f.read(2); return b, struct.unpack('<h', b)[0]
    elif vtype == GGUF_TYPE_UINT32:
        b = f.read(4); return b, struct.unpack('<I', b)[0]
    elif vtype == GGUF_TYPE_INT32:
        b = f.read(4); return b, struct.unpack('<i', b)[0]
    elif vtype == GGUF_TYPE_FLOAT32:
        b = f.read(4); return b, struct.unpack('<f', b)[0]
    elif vtype == GGUF_TYPE_BOOL:
        b = f.read(1); return b, bool(struct.unpack('<?', b)[0])
    elif vtype == GGUF_TYPE_STRING:
        s, raw = read_str_raw(f)
        return raw, s
    elif vtype == GGUF_TYPE_ARRAY:
        sub_type_bytes = f.read(4)
        sub_type = struct.unpack('<I', sub_type_bytes)[0]
        cnt_bytes = f.read(8)
        cnt = struct.unpack('<Q', cnt_bytes)[0]
        accum = [sub_type_bytes, cnt_bytes]
        items = []
        for _ in range(cnt):
            item_raw, item_val = read_val_raw(f, sub_type)
            accum.append(item_raw)
            if len(items) < 10:
                items.append(item_val)
        return b''.join(accum), items
    elif vtype == GGUF_TYPE_UINT64:
        b = f.read(8); return b, struct.unpack('<Q', b)[0]
    elif vtype == GGUF_TYPE_INT64:
        b = f.read(8); return b, struct.unpack('<q', b)[0]
    elif vtype == GGUF_TYPE_FLOAT64:
        b = f.read(8); return b, struct.unpack('<d', b)[0]
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


class GGUFInspector:
    def __init__(self, file_path: str):
        self.path = file_path
        self.kv_dict = {}
        self.kv_raw_list = []
        self.tensors = []
        self.arch = "unknown"
        self.block_count = 0
        self.alignment = GGUF_DEFAULT_ALIGNMENT
        self.data_start_offset = 0
        self._inspect()

    def _inspect(self):
        with open(self.path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF magic: {hex(magic)}")
            self.version = struct.unpack('<I', f.read(4))[0]
            self.n_tensors = struct.unpack('<Q', f.read(8))[0]
            self.n_kv = struct.unpack('<Q', f.read(8))[0]

            for _ in range(self.n_kv):
                key, vtype, raw_bytes, parsed_val = read_kv_pair_raw(f)
                self.kv_dict[key] = (vtype, parsed_val, raw_bytes)
                self.kv_raw_list.append((key, vtype, raw_bytes))

            self.arch = self.kv_dict.get("general.architecture", (None, "unknown"))[1]
            block_count_key = f"{self.arch}.block_count"
            if block_count_key in self.kv_dict:
                self.block_count = self.kv_dict[block_count_key][1]
            elif "qwen2.block_count" in self.kv_dict:
                self.block_count = self.kv_dict["qwen2.block_count"][1]
            elif "qwen35.block_count" in self.kv_dict:
                self.block_count = self.kv_dict["qwen35.block_count"][1]

            if "general.alignment" in self.kv_dict:
                self.alignment = self.kv_dict["general.alignment"][1]

            # Read tensor metadata
            for i in range(self.n_tensors):
                t_name, _ = read_str_raw(f)
                n_dims = struct.unpack('<I', f.read(4))[0]
                dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
                dtype = struct.unpack('<I', f.read(4))[0]
                offset_rel = struct.unpack('<Q', f.read(8))[0]

                # Compute byte size
                n_elems = 1
                for d in dims:
                    n_elems *= d

                if dtype in GGML_QUANT_SIZES:
                    bs, ts = GGML_QUANT_SIZES[dtype]
                    n_bytes = (n_elems * ts) // bs
                else:
                    n_bytes = n_elems * 2  # fallback

                self.tensors.append({
                    "index": i,
                    "name": t_name,
                    "n_dims": n_dims,
                    "dims": dims,
                    "dtype": dtype,
                    "offset_rel": offset_rel,
                    "n_bytes": n_bytes,
                })

            header_end = f.tell()
            self.data_start_offset = pad(header_end, self.alignment)


def prune_gguf_layers(input_path: str, output_path: str, layers_to_drop: list[int]):
    print(f"\n{'='*70}")
    print(f"SAMI SURGICAL GGUF PRUNER")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Layers to Drop: {layers_to_drop} ({len(layers_to_drop)} layers)")
    print(f"{'='*70}\n")

    t0 = time.time()
    inspector = GGUFInspector(input_path)
    print(f"Architecture:       {inspector.arch}")
    print(f"Original Layers:    {inspector.block_count}")
    print(f"Original Tensors:   {inspector.n_tensors}")
    print(f"Alignment:          {inspector.alignment}")
    print(f"Data Start Offset:  {inspector.data_start_offset} bytes")

    drop_set = set(layers_to_drop)
    new_block_count = inspector.block_count - len(layers_to_drop)
    print(f"New Layer Count:    {new_block_count}")

    # Build layer remapping
    kept_blocks = [b for b in range(inspector.block_count) if b not in drop_set]
    block_remap = {old_b: new_b for new_b, old_b in enumerate(kept_blocks)}

    # Filter and rename tensors
    filtered_tensors = []
    for t in inspector.tensors:
        name = t["name"]
        if name.startswith("blk."):
            parts = name.split(".")
            old_b = int(parts[1])
            if old_b in drop_set:
                continue
            new_b = block_remap[old_b]
            new_name = f"blk.{new_b}." + ".".join(parts[2:])
            t_copy = dict(t)
            t_copy["new_name"] = new_name
            filtered_tensors.append(t_copy)
        else:
            t_copy = dict(t)
            t_copy["new_name"] = name
            filtered_tensors.append(t_copy)

    print(f"Retained Tensors:   {len(filtered_tensors)} (Dropped {len(inspector.tensors) - len(filtered_tensors)})")

    # Build updated KV pairs
    updated_kv_raw = []
    arch_block_key = f"{inspector.arch}.block_count"
    
    for key, vtype, raw_bytes in inspector.kv_raw_list:
        if key in (arch_block_key, "qwen2.block_count", "qwen35.block_count"):
            # Update block_count
            key_raw = write_str(key)
            vtype_raw = struct.pack('<I', GGUF_TYPE_UINT32)
            val_raw = struct.pack('<I', new_block_count)
            updated_kv_raw.append(key_raw + vtype_raw + val_raw)
        else:
            updated_kv_raw.append(raw_bytes)

    # Compute new tensor offsets
    current_rel_offset = 0
    for t in filtered_tensors:
        t["new_offset_rel"] = current_rel_offset
        current_rel_offset += pad(t["n_bytes"], inspector.alignment)

    # Prepare Tensor Info binary section
    ti_bytes_list = []
    for t in filtered_tensors:
        ti_bytes_list.append(write_str(t["new_name"]))
        ti_bytes_list.append(struct.pack('<I', t["n_dims"]))
        for d in t["dims"]:
            ti_bytes_list.append(struct.pack('<Q', d))
        ti_bytes_list.append(struct.pack('<I', t["dtype"]))
        ti_bytes_list.append(struct.pack('<Q', t["new_offset_rel"]))

    ti_bytes = b''.join(ti_bytes_list)

    # Compute new file header
    header_magic = struct.pack('<I', GGUF_MAGIC)
    header_version = struct.pack('<I', inspector.version)
    header_n_tensors = struct.pack('<Q', len(filtered_tensors))
    header_n_kv = struct.pack('<Q', len(updated_kv_raw))

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nWriting pruned GGUF file...")
    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        # Write header
        fout.write(header_magic)
        fout.write(header_version)
        fout.write(header_n_tensors)
        fout.write(header_n_kv)

        # Write KV data
        for kv_data in updated_kv_raw:
            fout.write(kv_data)

        # Write TI data
        fout.write(ti_bytes)

        # Pad to alignment before tensor data
        header_size = fout.tell()
        data_aligned_start = pad(header_size, inspector.alignment)
        pad_needed = data_aligned_start - header_size
        if pad_needed > 0:
            fout.write(b'\x00' * pad_needed)

        print(f"Header written ({fout.tell()} bytes). Streaming tensor data...")

        # Stream tensor data from fin
        chunk_size = 64 * 1024 * 1024  # 64 MB
        total_tensors = len(filtered_tensors)
        written_bytes = 0

        for i, t in enumerate(filtered_tensors):
            old_abs_offset = inspector.data_start_offset + t["offset_rel"]
            fin.seek(old_abs_offset)
            
            bytes_to_copy = t["n_bytes"]
            while bytes_to_copy > 0:
                n = min(bytes_to_copy, chunk_size)
                buf = fin.read(n)
                if not buf:
                    raise IOError(f"Unexpected EOF reading tensor {t['name']} at {old_abs_offset}")
                fout.write(buf)
                bytes_to_copy -= len(buf)
                written_bytes += len(buf)

            # Pad tensor to alignment
            tensor_pad = pad(t["n_bytes"], inspector.alignment) - t["n_bytes"]
            if tensor_pad > 0:
                fout.write(b'\x00' * tensor_pad)

            if (i + 1) % 50 == 0 or (i + 1) == total_tensors:
                mb = written_bytes / (1024 * 1024)
                print(f"  [{i+1}/{total_tensors}] tensors written ({mb:.1f} MB)...")

    final_size = os.path.getsize(output_path)
    orig_size = os.path.getsize(input_path)
    savings = (1 - final_size / orig_size) * 100
    elapsed = time.time() - t0

    print(f"\nPruning Complete!")
    print(f"Original Size: {orig_size / (1024*1024*1024):.2f} GB")
    print(f"Pruned Size:   {final_size / (1024*1024*1024):.2f} GB ({savings:.1f}% reduction)")
    print(f"Elapsed Time:  {elapsed:.1f}s")
    print(f"Saved to:      {output_path}\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="SAMI GGUF Layer Pruner")
    parser.add_argument("--model", type=str, required=True, help="Input GGUF model path")
    parser.add_argument("--output", type=str, default=None, help="Output GGUF model path")
    parser.add_argument("--drop-layers", type=str, default="", help="Comma-separated list of layer indices to drop (e.g. 12,13,14,15)")
    parser.add_argument("--drop-periods", type=int, default=0, help="For models with periodic attention (e.g. Qwen3.5), drop N periods from the center")
    parser.add_argument("--info-only", action="store_true", help="Print model architecture and block structure only")
    args = parser.parse_args()

    inspector = GGUFInspector(args.model)
    print(f"\n--- Model Inspection: {args.model} ---")
    print(f"Architecture:       {inspector.arch}")
    print(f"Block Count:        {inspector.block_count}")
    print(f"Total Tensors:      {inspector.n_tensors}")

    full_attn_interval = 1
    full_attn_key = f"{inspector.arch}.full_attention_interval"
    if full_attn_key in inspector.kv_dict:
        full_attn_interval = inspector.kv_dict[full_attn_key][1]
        print(f"Attention Interval: Every {full_attn_interval} layers")

    if args.info_only:
        return

    layers_to_drop = []
    if args.drop_layers:
        layers_to_drop = [int(x.strip()) for x in args.drop_layers.split(",") if x.strip()]
    elif args.drop_periods > 0:
        interval = full_attn_interval if full_attn_interval > 1 else 4
        total_periods = inspector.block_count // interval
        print(f"Total Periods: {total_periods} (Period size = {interval} layers)")
        # Pick periods around the center: e.g. if 8 periods, center periods are 3, 4
        # Center index:
        center_period = total_periods // 2
        periods_to_drop = []
        for p in range(args.drop_periods):
            # Alternate around center
            if p % 2 == 0:
                p_idx = center_period + (p // 2)
            else:
                p_idx = center_period - ((p + 1) // 2)
            periods_to_drop.append(p_idx)
        
        periods_to_drop.sort()
        print(f"Dropping Period(s): {periods_to_drop}")
        for p_idx in periods_to_drop:
            start_l = p_idx * interval
            layers_to_drop.extend(range(start_l, start_l + interval))

    if not layers_to_drop:
        print("Error: No layers specified to drop. Use --drop-layers or --drop-periods.")
        sys.exit(1)

    layers_to_drop.sort()
    if args.output is None:
        p_name = Path(args.model).stem
        args.output = f"models/{p_name}_pruned_{inspector.block_count - len(layers_to_drop)}L.gguf"

    prune_gguf_layers(args.model, args.output, layers_to_drop)

if __name__ == "__main__":
    main()
