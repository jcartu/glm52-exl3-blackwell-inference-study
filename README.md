# GLM-5.2 EXL3 Blackwell Inference Study

A reproducible record of GLM-5.2 inference measurements collected on 23–24 July 2026 with four NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs. The study follows a rank-sliced 3.0 bpw EXL3/Trellis checkpoint from the original Gilded Gnosis v20 evaluation through an issue #34 RC2 rebase and an experimental EXL3-quantized MTP layer 78.

This is an engineering study, not a claim of a new model, quantization method, or general performance record. Results apply only to the pinned software, checkpoint variants, hardware topology, and benchmark settings recorded here.

> [!IMPORTANT]
> This repository contains study documentation, public-safe configurations, source patches, helper scripts, and measured artifacts. It does **not** contain or claim ownership of GLM-5.2 weights, the EXL3 checkpoint, vLLM, Sparkinfer, ExLlamaV3/Trellis, Gilded Gnosis, or referenced container images. See [CREDITS.md](CREDITS.md) for the complete ownership and contribution ledger.

## Current result

The selected profile combines:

- the exact issue #34 RC2 source pins;
- Brandon Music's EXL3 checkpoint and vLLM/Sparkinfer EXL3 work;
- David Young's dual-plan Trellis prefill work;
- a local rebase of that EXL3 integration onto RC2;
- targeted DCP workspace and required-tool grammar corrections;
- an offline EXL3 conversion of the previously BF16 MTP layer 78 using Brandon Music's published reproduction encoder as the base pipeline.

The runtime served with `TP4/DCP4/MTP3`, 3,072 maximum batched tokens, eight maximum sequences, and a configured 999,424-token model length. A direct smoke request containing **600,019 prompt tokens** completed and returned `CONTEXT_OK`. No 900K validation is claimed: the attempted 900K harness run was clamped to 128K by its parser and is excluded from publication.

### Cold prefill

Exact-token client measurements, one generated token:

| Prompt tokens | RC2+EXL3 merged baseline | BF16 MTP78 quality build | EXL3 MTP78 selected profile |
| ---: | ---: | ---: | ---: |
| 8K | 3,224 tok/s | 3,548 tok/s | **3,557 tok/s** |
| 64K | 1,782 tok/s | **2,075 tok/s** | 1,912 tok/s |
| 128K | 1,778 tok/s | **1,945 tok/s** | 1,857 tok/s |

The quantized MTP78 profile is not the fastest prefill profile at 64K or 128K. It was selected as the best measured balance of prefill, decode, memory reduction, quality, long-context operation, and tool behavior in this study.

### Sustained decode

Aggregate output tokens/second from steady-state windows:

| Input context | Profile | C1 | C2 | C4 | C8 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | RC2+EXL3 merged baseline | 101.7 | 163.7 | 237.5 | **372.0** |
| 0 | BF16 MTP78 quality build | 100.8 | **166.9** | 244.8 | **372.7** |
| 0 | EXL3 MTP78 selected profile | 99.5 | 160.2 | **249.5** | 369.6 |
| 128K | RC2+EXL3 merged baseline | 102.4 | 159.7 | capacity-limited | capacity-limited |
| 128K | BF16 MTP78 quality build | 100.8 | 160.2 | capacity-limited | capacity-limited |
| 128K | EXL3 MTP78 selected profile | **105.2** | **164.3** | **235.9** | capacity-limited |

The selected profile improves the measured 128K capacity surface, but it does not dominate every zero-context cell.

### Quality and behavior gates

LAVD reports exact / near / fail counts over ten deterministic runs:

| Profile | LAVD | Estonia | Required-tool gate |
| --- | ---: | ---: | --- |
| RC2+EXL3 merged baseline | 3 / 5 / 2 | not rerun | not rerun |
| BF16 MTP78 quality build | 4 / 5 / 1 | 10/10 | passed |
| EXL3 MTP78 selected profile | 6 / 2 / 2 | 10/10 | passed |

The selected profile produced two more exact LAVD answers than the BF16 control but also one additional hard failure in this ten-run sample. That mixed result is reported directly rather than summarized as an unqualified quality improvement.

The uncorrected required-tool path emitted 24 calls—23 duplicate Paris calls followed by `{}`. The corrected grammar emitted exactly one valid call and continued normally. This repository describes that as a targeted structured-output correction, not a general tool-calling result.

### MTP layer 78 conversion

The layer-78 conversion replaced 768 BF16 routed-expert projections with 12,288 EXL3 tensors covering all 256 experts and four TP ranks. The capture retained 131,072 finite rows. The assembled checkpoint reduced stored payload by **15,662,567,424 bytes total**, approximately 3.65 GiB per TP rank.

The conversion pipeline is derivative engineering around Brandon Music's published GLM-5.2 EXL3 reproduction bundle and ExLlamaV3/Trellis machinery. This study does not claim authorship of LDLQ, Trellis, MCG, EXL3, or the original checkpoint encoder.

## Attribution at a glance

- **Base model:** [Z.ai / GLM team](https://huggingface.co/zai-org/GLM-5.2).
- **EXL3 checkpoint and core integration:** [Brandon Music (`@brandonmmusic-max`)](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw), [vLLM PR #139](https://github.com/local-inference-lab/vllm/pull/139), and [Sparkinfer PR #49](https://github.com/local-inference-lab/sparkinfer/pull/49).
- **Dual-plan Trellis prefill:** [David Young (`@davidsyoung`), vLLM PR #163](https://github.com/local-inference-lab/vllm/pull/163).
- **Issue #34 and RC2 runtime work:** [Martin Vit (`@voipmonitor`)](https://github.com/voipmonitor), [`@yatesdr`](https://github.com/yatesdr), and the other contributors listed in [CREDITS.md](CREDITS.md).
- **Structured-output foundation:** [Florian Bernd (`@flobernd`), vLLM PR #34](https://github.com/local-inference-lab/vllm/pull/34).
- **Runtime foundations:** [vLLM](https://github.com/vllm-project/vllm), [Local Inference Lab](https://github.com/local-inference-lab), [Sparkinfer](https://github.com/local-inference-lab/sparkinfer), and [ExLlamaV3](https://github.com/turboderp-org/exllamav3) contributors.
- **Benchmark harness:** [Local Inference Lab `llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench), commit `86cf05c2f42f4d21b909b6e684424ca1aab89fd5`, reported version `0.4.29`.
- **Hardware, study direction, operation, measurements, and publication:** [Josh Cartu (`@jcartu`)](https://github.com/jcartu).
- **Implementation assistance:** OpenAI Codex; upstream PRs also disclose Anthropic Claude Code assistance and CodeRabbit review where applicable.

## Runtime pins

Issue #34 RC2 source image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm7e3bee1-si6234185-fi801d57a-cu132-20260723@sha256:67b17855ea81ebc8c9d7fc7c27d0d542c622347cd2607f0cf179e7cc4af2c1f0
```

Recorded source revisions:

- vLLM: `7e3bee1ed4bc...`
- Sparkinfer: `62341856cc54...`
- FlashInfer: `801d57...`
- issue #34 launcher source: `146fa...`

The final RC2+EXL3 image and MTP78 checkpoint were assembled locally and are not published as pullable images or redistributed weights. Reproduction therefore requires the upstream weights, the pinned RC2 source image, the patches in [`patches/`](patches/), and the MTP78 helper scripts in [`tools/mtp78/`](tools/mtp78/).

## Reproduce the serving profile

```bash
export MODEL_DIR="$HOME/models/GLM-5.2-EXL3-TR3-3.0bpw-MTP78"
export EXL3_IMAGE="your-local-rc2-exl3-image"
export CUDA_VISIBLE_DEVICES="3,1,2,0"  # machine-specific

docker compose \
  -f configs/docker-compose.rc2-exl3-quality.yml \
  up glm52-rc2-exl3-quality
```

The Compose file intentionally requires explicit local image and model paths. It does not pretend that the local build is available from a public registry.

## Repository map

```text
README.md                         current findings and claim boundaries
FOLLOWUP_STUDY.md                chronological follow-up analysis
METHODOLOGY.md                   original and RC2 benchmark methodology
CREDITS.md                       ownership, contribution, and license ledger
configs/                         public-safe measured serving configurations
patches/                         source patches applied to pinned RC2 trees
tools/mtp78/                     capture and checkpoint-assembly glue
results/raw/                     original and RC2+EXL3 raw measurements
results/followup/                renamed July 23 follow-up evidence
results/issue34/                 issue #34 RC2 comparison evidence
results/*-manifest.json          artifact provenance, sizes, and hashes
results/SHA256SUMS               complete publication integrity index
```

Historical raw output filenames containing `winning` are retained byte-for-byte because they are original benchmark output names. The study text treats them only as selected historical profiles.

## Integrity and scope

Verify the publication from the repository root:

```bash
sha256sum --check results/SHA256SUMS
```

Raw evidence includes hostname, driver, PCI topology, and local endpoint values where they affect reproducibility. It contains no model weights, container layers, credentials, API keys, or private network endpoints. See [results/README.md](results/README.md) for the artifact index and exclusions.

## License

Original study documentation, public-safe configurations, helper scripts, and generated metadata are released under the [MIT License](LICENSE). Referenced models, runtimes, libraries, source patches, and container components retain their upstream licenses and copyright notices. See [CREDITS.md](CREDITS.md) before redistributing third-party material.
