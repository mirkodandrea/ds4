#!/usr/bin/env python3
"""Split a ds4 GGUF into coordinator (non-expert) and expert-only files.

Usage:
    python split_gguf.py ds4flash.gguf

Produces:
    ds4flash-coord.gguf   — attention, routing, shared experts, embeddings
    ds4flash-experts.gguf  — routed expert weights only (ffn_{gate,up,down}_exps)
"""

import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

EXPERT_TENSOR_SUFFIXES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)

def is_expert_tensor(name: str) -> bool:
    return any(name.endswith(s) for s in EXPERT_TENSOR_SUFFIXES)


def read_field_value(field):
    """Extract a Python value from a GGUFReader field using its data indices."""
    from gguf import GGUFValueType

    if not field.types:
        return None

    first_type = field.types[0]

    if first_type == GGUFValueType.ARRAY:
        if len(field.types) < 2:
            return None
        elem_type = field.types[1]
        values = []
        for idx in field.data:
            part = field.parts[idx]
            if elem_type == GGUFValueType.STRING:
                values.append(bytes(part).decode("utf-8"))
            else:
                values.append(part[0])

        if elem_type == GGUFValueType.STRING:
            return values
        # Return as numpy array with correct dtype
        dtype_map = {
            GGUFValueType.UINT8: np.uint8,
            GGUFValueType.INT8: np.int8,
            GGUFValueType.UINT16: np.uint16,
            GGUFValueType.INT16: np.int16,
            GGUFValueType.UINT32: np.uint32,
            GGUFValueType.INT32: np.int32,
            GGUFValueType.UINT64: np.uint64,
            GGUFValueType.INT64: np.int64,
            GGUFValueType.FLOAT32: np.float32,
            GGUFValueType.FLOAT64: np.float64,
            GGUFValueType.BOOL: np.bool_,
        }
        dt = dtype_map.get(elem_type)
        if dt is None:
            return None
        return np.array(values, dtype=dt)

    # Scalar — f.data has one index pointing to the value part
    val_part = field.parts[field.data[0]]
    if first_type == GGUFValueType.STRING:
        return bytes(val_part).decode("utf-8")
    return val_part[0]


def write_field_to_writer(writer, field_name, field):
    """Write a reader field to a GGUFWriter."""
    from gguf import GGUFValueType

    if not field.types:
        return

    first_type = field.types[0]
    val = read_field_value(field)
    if val is None:
        print(f"  Skipping unsupported field: {field_name}")
        return

    if first_type == GGUFValueType.ARRAY:
        writer.add_array(field_name, val)
    elif first_type == GGUFValueType.STRING:
        writer.add_string(field_name, val)
    elif first_type == GGUFValueType.UINT8:
        writer.add_uint8(field_name, int(val))
    elif first_type == GGUFValueType.INT8:
        writer.add_int8(field_name, int(val))
    elif first_type == GGUFValueType.UINT16:
        writer.add_uint16(field_name, int(val))
    elif first_type == GGUFValueType.INT16:
        writer.add_int16(field_name, int(val))
    elif first_type == GGUFValueType.UINT32:
        writer.add_uint32(field_name, int(val))
    elif first_type == GGUFValueType.INT32:
        writer.add_int32(field_name, int(val))
    elif first_type == GGUFValueType.UINT64:
        writer.add_uint64(field_name, int(val))
    elif first_type == GGUFValueType.INT64:
        writer.add_int64(field_name, int(val))
    elif first_type == GGUFValueType.FLOAT32:
        writer.add_float32(field_name, float(val))
    elif first_type == GGUFValueType.FLOAT64:
        writer.add_float64(field_name, float(val))
    elif first_type == GGUFValueType.BOOL:
        writer.add_bool(field_name, bool(val))
    else:
        print(f"  Skipping unsupported type: {field_name} type={first_type}")


def split_gguf(input_path: str) -> None:
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

    inp = Path(input_path)
    stem = inp.stem
    coord_path = inp.with_name(f"{stem}-coord.gguf")
    expert_path = inp.with_name(f"{stem}-experts.gguf")

    print(f"Reading {inp} ...")
    reader = GGUFReader(str(inp))

    coord_tensors = [t for t in reader.tensors if not is_expert_tensor(t.name)]
    expert_tensors = [t for t in reader.tensors if is_expert_tensor(t.name)]

    coord_bytes = sum(t.n_bytes for t in coord_tensors)
    expert_bytes = sum(t.n_bytes for t in expert_tensors)
    print(f"Coordinator: {len(coord_tensors)} tensors, {coord_bytes / 2**30:.2f} GiB")
    print(f"Experts:     {len(expert_tensors)} tensors, {expert_bytes / 2**30:.2f} GiB")

    arch = "deepseek4"

    def write_split(out_path: Path, tensors, label: str):
        print(f"\nWriting {label}: {out_path}")
        writer = GGUFWriter(str(out_path), arch)

        # Copy all metadata from source
        for field_name, field in reader.fields.items():
            if field_name.startswith("GGUF."):
                continue
            if field_name == "general.architecture":
                continue  # already set by GGUFWriter constructor
            write_field_to_writer(writer, field_name, field)

        # Write tensors
        total_bytes = sum(t.n_bytes for t in tensors)
        written = 0
        pbar = tqdm(tensors, desc=f"  Tensors", unit="tensor")
        for t in pbar:
            raw_dtype = GGMLQuantizationType(t.tensor_type)
            raw_shape = list(reversed(t.shape))  # GGUF stores dims in reverse
            writer.add_tensor(t.name, t.data, raw_shape=raw_shape, raw_dtype=raw_dtype)
            written += t.n_bytes
            pbar.set_postfix(GiB=f"{written / 2**30:.1f}/{total_bytes / 2**30:.1f}")

        print(f"  Flushing to disk...")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        print(f"  Done: {os.path.getsize(str(out_path)) / 2**30:.2f} GiB")

    write_split(coord_path, coord_tensors, "coordinator")
    write_split(expert_path, expert_tensors, "experts")

    print(f"\n✓ Split complete:")
    print(f"  {coord_path.name}  — load on coordinator (Mac)")
    print(f"  {expert_path.name} — load on expert server (fuoco)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input.gguf>")
        sys.exit(1)
    split_gguf(sys.argv[1])
