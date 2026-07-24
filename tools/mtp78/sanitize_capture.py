#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

HIDDEN = 6144
TOPK = 8


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove non-finite BF16 rows from an MTP capture")
    parser.add_argument("--layer-dir", required=True, type=Path)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()
    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows must be positive")

    x_path = args.layer_dir / "x.bin"
    ids_path = args.layer_dir / "ids.bin"
    rows = x_path.stat().st_size // (HIDDEN * 2)
    if rows < 1 or x_path.stat().st_size != rows * HIDDEN * 2:
        raise RuntimeError("x.bin is empty or unaligned")
    if ids_path.stat().st_size != rows * TOPK:
        raise RuntimeError("ids.bin row count does not match x.bin")

    x = np.memmap(x_path, mode="r", dtype=np.uint16, shape=(rows, HIDDEN))
    ids = np.memmap(ids_path, mode="r", dtype=np.uint8, shape=(rows, TOPK))
    x_tmp = x_path.with_suffix(".bin.finite")
    ids_tmp = ids_path.with_suffix(".bin.finite")
    kept = 0
    with x_tmp.open("wb", buffering=0) as x_out, ids_tmp.open("wb", buffering=0) as ids_out:
        for start in range(0, rows, args.chunk_rows):
            stop = min(start + args.chunk_rows, rows)
            block = np.asarray(x[start:stop])
            finite = np.all((block & np.uint16(0x7F80)) != np.uint16(0x7F80), axis=1)
            x_out.write(block[finite].tobytes())
            ids_out.write(np.asarray(ids[start:stop])[finite].tobytes())
            kept += int(finite.sum())
        os.fsync(x_out.fileno())
        os.fsync(ids_out.fileno())
    del x, ids

    if x_tmp.stat().st_size != kept * HIDDEN * 2 or ids_tmp.stat().st_size != kept * TOPK:
        raise RuntimeError("compacted payload size mismatch")
    os.replace(x_tmp, x_path)
    os.replace(ids_tmp, ids_path)
    complete = args.layer_dir / "COMPLETE"
    if complete.exists():
        complete.unlink()
    manifest = args.layer_dir / "layer_manifest.json"
    if manifest.exists():
        manifest.unlink()
    print(json.dumps({"input_rows": rows, "kept_rows": kept, "removed_rows": rows - kept}, sort_keys=True))


if __name__ == "__main__":
    main()
