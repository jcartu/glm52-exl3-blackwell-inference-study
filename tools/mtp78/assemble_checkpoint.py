#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

_EXPERT_WEIGHT = re.compile(
    r"^model\.layers\.78\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)\.weight$"
)
_EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(64 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reflink_copy(source: str, destination: str) -> str:
    """Clone a file without sharing its inode so the graft can be read-only."""
    try:
        with open(source, "rb") as source_file, open(destination, "xb") as output_file:
            fcntl.ioctl(output_file.fileno(), 0x40049409, source_file.fileno())
    except OSError as error:
        Path(destination).unlink(missing_ok=True)
        raise RuntimeError(
            f"reflink clone failed for {source}; refusing a full copy or hardlink"
        ) from error
    shutil.copystat(source, destination)
    return destination


def load_encoder(path: Path):
    spec = importlib.util.spec_from_file_location("_mtp78_encoder_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import encoder helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate quantized GLM-5.2 MTP layer 78")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--encoded-layer", required=True, type=Path)
    parser.add_argument("--done", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--encoder-base",
        type=Path,
        default=Path(__file__).with_name("encoder") / "encode_tr3_v31.py",
    )
    parser.add_argument("--expected-encoded-sha256")
    parser.add_argument("--source-revision")
    parser.add_argument("--overlay-repository")
    parser.add_argument("--overlay-revision")
    parser.add_argument("--overlay-variant")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    encoded_sha256 = sha256_file(args.encoded_layer)
    if (
        args.expected_encoded_sha256 is not None
        and encoded_sha256 != args.expected_encoded_sha256
    ):
        raise RuntimeError(
            "encoded layer SHA-256 mismatch: "
            f"actual={encoded_sha256} expected={args.expected_encoded_sha256}"
        )

    base = load_encoder(args.encoder_base.resolve())
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    layer_files = {
        filename
        for name, filename in weight_map.items()
        if name.startswith("model.layers.78.")
    }
    if len(layer_files) != 1:
        raise RuntimeError(f"layer 78 spans unexpected shards: {sorted(layer_files)}")
    layer_file = next(iter(layer_files))
    skip = {
        "config.json",
        "model.safetensors.index.json",
        "tier_bitmap.json",
        "MANIFEST.sha256",
        layer_file,
    }

    def ignore_top_level(directory: str, names: list[str]) -> set[str]:
        return skip.intersection(names) if Path(directory).resolve() == source else set()

    shutil.copytree(source, output, copy_function=reflink_copy, ignore=ignore_top_level)

    source_reader = base.STReader(str(source / layer_file))
    encoded_reader = base.STReader(str(args.encoded_layer))
    dropped = sorted(name for name in source_reader.tensors if _EXPERT_WEIGHT.match(name))
    if len(dropped) != 256 * 3:
        raise RuntimeError(f"expected 768 BF16 MTP expert weights, found {len(dropped)}")
    if len(encoded_reader.tensors) != 256 * 3 * 4 * 4:
        raise RuntimeError(
            f"expected 12288 EXL3 MTP tensors, found {len(encoded_reader.tensors)}"
        )

    done = json.loads(args.done.read_text())
    expected_encoded = {
        f"model.layers.78.mlp.experts.{expert}.{projection}.rank{rank}.{suffix}"
        for expert in range(256)
        for projection in base.PROJS
        for rank in range(4)
        for suffix in _EXL3_SUFFIXES
    }
    if set(encoded_reader.tensors) != expected_encoded:
        missing = sorted(expected_encoded - set(encoded_reader.tensors))[:4]
        extra = sorted(set(encoded_reader.tensors) - expected_encoded)[:4]
        raise RuntimeError(f"EXL3 layer schema mismatch: missing={missing}, extra={extra}")
    if done.get("schema") not in (None, "glm52-mtp78-exl3-layer-v1"):
        raise RuntimeError(f"unsupported encoded layer schema: {done['schema']}")
    done_checks = (
        int(done.get("layer", -1)) == 78,
        int(done.get("bits", -1)) == 3,
        int(done.get("tp", -1)) == 4,
        done.get("keep_nvfp4") == [],
        done.get("tail_tr3") == list(range(256)),
        int(done.get("tensor_count", len(expected_encoded))) == len(expected_encoded),
        int(done.get("source_expert_tensor_count", len(dropped))) == len(dropped),
        int(done.get("slices_with_out_scales", 256 * 3 * 4)) == 256 * 3 * 4,
        done.get("file_sha256") == encoded_sha256,
    )
    if not all(done_checks):
        raise RuntimeError("encoded layer done artifact is stale or invalid")

    source_digest = hashlib.sha256()
    for expert in range(256):
        for projection in base.PROJS:
            key = base.expert_key(78, expert, projection, "weight")
            source_digest.update(key.encode())
            source_digest.update(b"\0")
            source_digest.update(source_reader.read_bytes(key))
    source_expert_payload_sha256 = source_digest.hexdigest()
    published_source_sha256 = done.get("source_expert_payload_sha256")
    if (
        published_source_sha256 is not None
        and source_expert_payload_sha256 != published_source_sha256
    ):
        raise RuntimeError("source MTP expert payload changed after quantization")
    done["schema"] = "glm52-mtp78-exl3-layer-v1"
    done["tensor_count"] = len(expected_encoded)
    done["source_expert_tensor_count"] = len(dropped)
    done["source_expert_payload_sha256"] = source_expert_payload_sha256

    entries = []
    for name in sorted(source_reader.tensors):
        if name in dropped:
            continue
        dtype, shape, _, _ = source_reader.tensors[name]
        entries.append(
            (name, dtype, shape, lambda name=name: source_reader.read_bytes(name))
        )
    for name in sorted(encoded_reader.tensors):
        dtype, shape, _, _ = encoded_reader.tensors[name]
        entries.append(
            (name, dtype, shape, lambda name=name: encoded_reader.read_bytes(name))
        )
    base.write_safetensors(str(output / layer_file), entries, metadata={"format": "pt"})

    new_weight_map = {name: filename for name, filename in weight_map.items() if name not in dropped}
    new_weight_map.update({name: layer_file for name in encoded_reader.tensors})
    old_total = int(index.get("metadata", {}).get("total_size", 0))
    removed_bytes = sum(source_reader.nbytes(name) for name in dropped)
    added_bytes = sum(encoded_reader.nbytes(name) for name in encoded_reader.tensors)
    index["weight_map"] = dict(sorted(new_weight_map.items()))
    index.setdefault("metadata", {})["total_size"] = old_total - removed_bytes + added_bytes
    atomic_json(output / "model.safetensors.index.json", index)

    config = json.loads((source / "config.json").read_text())
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict) or not isinstance(
        quantization.get("ignore"), list
    ):
        raise RuntimeError("source quantization_config ignore list is missing")
    if "model.layers.78*" not in quantization["ignore"]:
        raise RuntimeError("source quantization_config does not isolate BF16 MTP layer 78")
    quantization["ignore"] = [
        item for item in quantization["ignore"] if item != "model.layers.78*"
    ]
    hybrid = config["hybrid_tr3_tail"]
    hybrid["moe_layers"] = [3, 78]
    hybrid["tr3_tail_per_layer"] = 256
    hybrid["modelopt_dispatch_note"] = (
        "quantization_config selects the b12x ModelOpt interception; routed experts "
        "in layers 3..78 are EXL3, including the MTP draft layer."
    )
    scope = hybrid.get("scope", {})
    if isinstance(scope.get("bf16_byte_exact"), list):
        scope["bf16_byte_exact"] = [
            item for item in scope["bf16_byte_exact"] if item != "MTP layer 78"
        ]
    scope["quantized"] = (
        "routed MoE expert gate/up/down projections, all 256 experts, layers 3..78"
    )
    hybrid["mtp_layer_78"] = {
        "hessian": done.get("hessian"),
        "capture": done.get("capture"),
        "recipe_fingerprint": done.get("recipe_fingerprint"),
        "source_expert_payload_sha256": source_expert_payload_sha256,
        "file_sha256": encoded_sha256,
        "overlay_repository": args.overlay_repository,
        "overlay_revision": args.overlay_revision,
        "overlay_variant": args.overlay_variant,
        "source_revision": args.source_revision,
    }
    atomic_json(output / "config.json", config)

    tier_path = source / "tier_bitmap.json"
    if tier_path.exists():
        tier_bitmap = json.loads(tier_path.read_text())
        tier_bitmap["78"] = done
        atomic_json(output / "tier_bitmap.json", tier_bitmap)

    source_manifest = source / "MANIFEST.sha256"
    provenance = {
        "schema": "glm52-mtp78-graft-v1",
        "source": {
            "path": str(source),
            "revision": args.source_revision,
            "config_sha256": sha256_file(source / "config.json"),
            "index_sha256": sha256_file(source / "model.safetensors.index.json"),
            "manifest_sha256": (
                sha256_file(source_manifest) if source_manifest.exists() else None
            ),
            "expert_payload_sha256": source_expert_payload_sha256,
        },
        "overlay": {
            "path": str(args.encoded_layer.resolve()),
            "repository": args.overlay_repository,
            "revision": args.overlay_revision,
            "variant": args.overlay_variant,
            "sha256": encoded_sha256,
            "metadata_sha256": sha256_file(args.done),
        },
        "integration": {
            "layer_file": layer_file,
            "dropped_bf16_weights": len(dropped),
            "added_exl3_tensors": len(encoded_reader.tensors),
            "carried_non_expert_tensors": len(source_reader.tensors) - len(dropped),
            "removed_bytes": removed_bytes,
            "added_bytes": added_bytes,
        },
    }
    atomic_json(output / "GRAFT_PROVENANCE.json", provenance)
    manifest_lines = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(64 << 20), b""):
                digest.update(chunk)
        manifest_lines.append(f"{digest.hexdigest()}  {path.relative_to(output)}")
    (output / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n")

    print(
        json.dumps(
            {
                "output": str(output),
                "layer_file": layer_file,
                "dropped_bf16_weights": len(dropped),
                "added_exl3_tensors": len(encoded_reader.tensors),
                "removed_bytes": removed_bytes,
                "added_bytes": added_bytes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
