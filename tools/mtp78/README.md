# MTP Layer 78 Helper Scripts

These locally authored helpers capture routed activations from GLM-5.2 MTP layer 78, validate and seal the capture, drive requests, remove non-finite rows from interrupted captures, and assemble an EXL3-encoded layer into a copy-on-write checkpoint tree.

They are **glue around upstream work**, not a standalone quantizer. The actual LDLQ/Trellis/MCG encoder and tensor helpers come from Brandon Music's [GLM-5.2 EXL3 reproduction bundle](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw/tree/main/encoder), which in turn uses [ExLlamaV3](https://github.com/turboderp-org/exllamav3). Acquire and review those upstream sources and licenses separately.

## Files

| File | Role |
| --- | --- |
| `capture/sitecustomize.py` | Narrow runtime hook that records finite layer-78 hidden rows and routed expert IDs from TP rank 0 while an `ENABLE` sentinel exists. |
| `drive_capture.py` | Sends owner-supplied JSONL prompts until the configured capture limit is reached; supports controlled resume. |
| `sanitize_capture.py` | Removes non-finite BF16 rows and aligned expert IDs from an interrupted capture. |
| `finalize_capture.py` | Checks shape/routing invariants and writes a fingerprinted capture plan and manifest. |
| `assemble_checkpoint.py` | Validates the encoded layer, replaces only layer-78 BF16 expert projections, updates metadata, and emits `MANIFEST.sha256`. |

## Recorded run

The published run captured 131,072 finite rows with hidden size 6,144 and top-k 8 natural routing. It encoded all 256 experts into 12,288 TP4 EXL3 tensors and reduced stored checkpoint payload by 15,662,567,424 bytes total. See [`../../results/mtp78-capture-plan.json`](../../results/mtp78-capture-plan.json) and [`../../results/mtp78-conversion.json`](../../results/mtp78-conversion.json).

Capture and encoded payloads are intentionally absent because they derive from model activations and weights. The calibration corpus is also not redistributed.

## Capture outline

```bash
export EXL3_IMAGE="your-local-rc2-exl3-image"
export MODEL_DIR="$HOME/models/GLM-5.2-EXL3-TR3-3.0bpw"
export MTP_HOOK_DIR="$PWD/tools/mtp78/capture"
export MTP_CAPTURE_DIR="/dev/shm/glm52-mtp78-capture"

docker compose -f configs/docker-compose.mtp78-capture.yml up glm52-mtp78-capture

python tools/mtp78/drive_capture.py \
  --corpus /path/to/your.jsonl \
  --capture-layer-dir "$MTP_CAPTURE_DIR/layer_078"

python tools/mtp78/finalize_capture.py \
  --source "$MODEL_DIR" \
  --capture-dir "$MTP_CAPTURE_DIR" \
  --plan /tmp/capture-plan.json
```

Encoding requires adapting the upstream encoder to layer 78 and supplying the sealed capture. Assembly then uses:

```bash
python tools/mtp78/assemble_checkpoint.py \
  --source "$MODEL_DIR" \
  --encoded-layer /path/to/tr3-layer-078.safetensors \
  --done /path/to/layer-078.done.json \
  --encoder-base /path/to/upstream/encoder/encode_tr3_v31.py \
  --output "$HOME/models/GLM-5.2-EXL3-TR3-3.0bpw-MTP78"
```

The assembler refuses an existing output directory, unexpected tensor schemas, stale hashes, source payload drift, or a source checkpoint that does not explicitly isolate BF16 layer 78.
