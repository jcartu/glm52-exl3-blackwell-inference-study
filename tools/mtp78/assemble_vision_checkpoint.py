#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

from safetensors import safe_open

BF16_ASSETS = ("vision_tower.safetensors", "mm_projector.safetensors")
ASSET_FILES = (
    "chat_template.jinja",
    "configuration_glm5v.py",
    "generation_config.json",
    "kimi_k25_processor.py",
    "kimi_k25_vision_processing.py",
    "media_utils.py",
    "preprocessor_config.json",
    "tokenizer_config.json",
)
SOURCE_METADATA = (
    "GRAFT_PROVENANCE.json",
    "tier_bitmap.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(64 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reflink_copy(source: Path, destination: Path) -> None:
    try:
        with source.open("rb") as source_file, destination.open("xb") as output_file:
            fcntl.ioctl(output_file.fileno(), 0x40049409, source_file.fileno())
    except OSError as error:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"reflink clone failed for {source}; refusing a full copy or hardlink"
        ) from error
    shutil.copystat(source, destination)
    if source.stat().st_ino == destination.stat().st_ino:
        raise RuntimeError(f"reflink unexpectedly shares inode: {source}")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def bf16_tensor_inventory(path: Path) -> tuple[list[str], int]:
    keys: list[str] = []
    total_bytes = 0
    with safe_open(path, framework="pt", device="cpu") as file:
        for key in file.keys():
            tensor = file.get_slice(key)
            if str(tensor.get_dtype()) != "BF16":
                raise RuntimeError(f"non-BF16 vision tensor {key} in {path}")
            keys.append(key)
            elements = 1
            for dimension in tensor.get_shape():
                elements *= dimension
            total_bytes += elements * 2
    return keys, total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble an immutable GLM-5.2 EXL3 + BF16 MoonViT checkpoint"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vision-revision", required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    assets = args.assets.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    for required in (
        source / "config.json",
        source / "model.safetensors.index.json",
        source / "tokenizer.json",
        assets / "config.json",
        assets / "model.safetensors.index.json",
        *(assets / name for name in BF16_ASSETS),
        *(assets / name for name in ASSET_FILES),
    ):
        if not required.is_file():
            raise RuntimeError(f"required input is missing: {required}")

    source_config = json.loads((source / "config.json").read_text())
    asset_config = json.loads((assets / "config.json").read_text())
    if source_config.get("architectures") != ["GlmMoeDsaForCausalLM"]:
        raise RuntimeError("source is not a GLM-5.2 text checkpoint")
    if asset_config.get("architectures") != ["Glm5vForConditionalGeneration"]:
        raise RuntimeError("assets are not the pinned GLM5V checkpoint")
    if int(source_config.get("num_hidden_layers", -1)) != 78:
        raise RuntimeError("source text config does not contain 78 target layers")
    if int(source_config.get("num_nextn_predict_layers", -1)) != 1:
        raise RuntimeError("source text config does not expose MTP layer 78")

    inventories: dict[str, list[str]] = {}
    bf16_bytes = 0
    for name in BF16_ASSETS:
        inventories[name], tensor_bytes = bf16_tensor_inventory(assets / name)
        bf16_bytes += tensor_bytes

    output.mkdir(parents=True)
    try:
        source_files = sorted(source.glob("model*.safetensors"))
        if not source_files:
            raise RuntimeError("source contains no model safetensors")
        for source_file in source_files:
            reflink_copy(source_file, output / source_file.name)
        reflink_copy(source / "tokenizer.json", output / "tokenizer.json")
        for name in SOURCE_METADATA:
            path = source / name
            if path.is_file():
                reflink_copy(path, output / name)
        for name in BF16_ASSETS:
            reflink_copy(assets / name, output / name)
        for name in ASSET_FILES:
            shutil.copy2(assets / name, output / name)

        source_index = json.loads((source / "model.safetensors.index.json").read_text())
        asset_index = json.loads((assets / "model.safetensors.index.json").read_text())
        source_map = source_index["weight_map"]
        asset_map = asset_index["weight_map"]
        media_keys = inventories["vision_tower.safetensors"] + inventories[
            "mm_projector.safetensors"
        ]
        if any(key in source_map for key in media_keys):
            raise RuntimeError("vision tensor collides with source text tensor")
        for key in media_keys:
            filename = asset_map.get(key)
            if filename not in BF16_ASSETS:
                raise RuntimeError(f"vision index mismatch for {key}: {filename}")
            source_map[key] = filename
        source_index["weight_map"] = dict(sorted(source_map.items()))
        source_index.setdefault("metadata", {})["total_size"] = int(
            source_index.get("metadata", {}).get("total_size", 0)
        ) + bf16_bytes
        atomic_json(output / "model.safetensors.index.json", source_index)

        text_config = dict(source_config)
        text_config.setdefault("bos_token_id", 0)
        output_config = {
            key: value
            for key, value in asset_config.items()
            if key not in {"text_config", "quantization_config"}
        }
        vision_config = dict(output_config["vision_config"])
        vision_config["mm_hidden_size"] = text_config["hidden_size"]
        vision_config["text_hidden_size"] = text_config["hidden_size"]
        output_config["vision_config"] = vision_config
        output_config["text_config"] = text_config
        output_config["quantization_config"] = text_config["quantization_config"]
        output_config["vision_graft"] = {
            "schema": "glm52-exl3-glm5v-graft-v2",
            "variant": args.variant,
            "vision_repository": "baseten/GLM-5.2-Vision-NVFP4",
            "vision_revision": args.vision_revision,
            "vision_tower_dtype": "bfloat16",
            "mm_projector_dtype": "bfloat16",
            "text_source": str(source),
        }
        atomic_json(output / "config.json", output_config)

        asset_hashes = {
            name: sha256_file(assets / name)
            for name in (*BF16_ASSETS, "config.json", "preprocessor_config.json")
        }
        source_manifest = source / "MANIFEST.sha256"
        provenance = {
            "schema": "glm52-exl3-glm5v-graft-v2",
            "variant": args.variant,
            "source": {
                "path": str(source),
                "config_sha256": sha256_file(source / "config.json"),
                "index_sha256": sha256_file(source / "model.safetensors.index.json"),
                "manifest_sha256": sha256_file(source_manifest)
                if source_manifest.is_file()
                else None,
            },
            "vision": {
                "repository": "baseten/GLM-5.2-Vision-NVFP4",
                "revision": args.vision_revision,
                "asset_hashes": asset_hashes,
                "tensor_counts": {name: len(keys) for name, keys in inventories.items()},
                "tensor_bytes": bf16_bytes,
                "dtypes": ["BF16"],
            },
            "integration": {
                "private_reflinks": True,
                "outer_architecture": "Glm5vForConditionalGeneration",
                "text_architecture": "GlmMoeDsaForCausalLM",
                "target_layers": 78,
                "mtp_layers": 1,
                "media_placeholder_token_id": output_config[
                    "media_placeholder_token_id"
                ],
            },
        }
        atomic_json(output / "VISION_GRAFT_PROVENANCE.json", provenance)

        for root, directories, files in os.walk(output):
            for filename in files:
                os.chmod(Path(root) / filename, 0o444)
            for directory in directories:
                os.chmod(Path(root) / directory, 0o555)
        os.chmod(output, 0o555)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output": str(output),
                "variant": args.variant,
                "text_shards": len(source_files),
                "vision_tensors": sum(len(keys) for keys in inventories.values()),
                "vision_tensor_bytes": bf16_bytes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
