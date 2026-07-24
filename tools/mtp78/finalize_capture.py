#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HIDDEN = 6144
TOPK = 8
LAYER = 78
CORPUS_SHA256 = "cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(64 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal a live GLM-5.2 MTP layer capture")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()

    layer_dir = args.capture_dir / f"layer_{LAYER:03d}"
    x_path = layer_dir / "x.bin"
    ids_path = layer_dir / "ids.bin"
    x_bytes = x_path.stat().st_size
    ids_bytes = ids_path.stat().st_size
    if x_bytes % (HIDDEN * 2) or ids_bytes % TOPK:
        raise RuntimeError(f"unaligned capture payload: x={x_bytes}, ids={ids_bytes}")
    x_rows = x_bytes // (HIDDEN * 2)
    ids_rows = ids_bytes // TOPK
    if x_rows != ids_rows or x_rows == 0:
        raise RuntimeError(f"capture row mismatch: x={x_rows}, ids={ids_rows}")

    ids = np.memmap(ids_path, mode="r", dtype=np.uint8, shape=(ids_rows, TOPK))
    if int(ids.max()) >= 256:
        raise RuntimeError("capture contains an out-of-range expert id")
    sorted_ids = np.sort(ids, axis=1)
    if not np.all(sorted_ids[:, 1:] != sorted_ids[:, :-1]):
        raise RuntimeError("capture contains a duplicate routed expert within a token")
    routed_counts = np.bincount(ids.reshape(-1), minlength=256)

    config_path = args.source / "config.json"
    index_path = args.source / "model.safetensors.index.json"
    plan = {
        "schema": "glm52-mtp78-capture-plan-v1",
        "corpus_sha256": CORPUS_SHA256,
        "source": {
            "config_sha256": sha256_file(config_path),
            "index_sha256": sha256_file(index_path),
        },
        "capture_tp": 4,
        "output_tp": 4,
        "selection_policy": "mtp-live-draft-owner-corpus-v1",
        "owner_corpus_only": True,
        "calibration_baseline": True,
        "routing": {
            "natural": True,
            "forced_expert_activation": False,
            "scoring_func": "sigmoid",
            "top_k": TOPK,
            "n_group": 1,
            "topk_group": 1,
        },
        "total_tokens": x_rows,
    }
    plan["capture_fingerprint"] = canonical_hash(plan)
    manifest = {
        "layer": LAYER,
        "capture_fingerprint": plan["capture_fingerprint"],
        "tokens": x_rows,
        "hidden": HIDDEN,
        "x_dtype": "bfloat16",
        "ids_dtype": "uint8",
        "top_k": TOPK,
        "sha256_x": sha256_file(x_path),
        "sha256_ids": sha256_file(ids_path),
        "routed_counts": routed_counts.tolist(),
    }
    atomic_json(args.plan, plan)
    atomic_json(layer_dir / "layer_manifest.json", manifest)
    print(
        json.dumps(
            {
                "tokens": x_rows,
                "routed_min": int(routed_counts.min()),
                "routed_max": int(routed_counts.max()),
                "capture_fingerprint": plan["capture_fingerprint"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
