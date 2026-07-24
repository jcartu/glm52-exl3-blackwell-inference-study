# GLM-5.2 EXL3 Blackwell Inference Study

A reproducible record of the 23–24 July 2026 tuning campaigns for GLM-5.2 on four NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs, centered on the rank-sliced 3.0 bpw EXL3/Trellis build and followed by a quality-gated Gilded Gnosis v20 RC2 evaluation.

The campaign moved from the validated v2 baseline to the Gilded Gnosis v20 runtime, measured each consequential serving choice, retained rejected and failed candidates, and then applied performance, long-context, quality, tool-calling, power, and thermal gates to the winning configuration.

> [!IMPORTANT]
> This repository contains original benchmark configuration, methodology, and measured result artifacts. It does **not** contain or claim ownership of GLM-5.2 weights, the EXL3 checkpoint, vLLM, Sparkinfer, ExLlamaV3/Trellis, Gilded Gnosis, or the referenced container images. Those components remain the work of their respective authors and are linked throughout this document and in [CREDITS.md](CREDITS.md).

## Attribution at a glance

- **Base model:** [Z.ai, `zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2).
- **Rank-sliced EXL3 checkpoint and core EXL3 runtime integration:** [Brandon Music (`brandonmusic`, `@brandonmmusic-max`)](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw), including [vLLM PR #139](https://github.com/local-inference-lab/vllm/pull/139) and [Sparkinfer PR #49](https://github.com/local-inference-lab/sparkinfer/pull/49).
- **Dual-plan Trellis prefill:** [David Young (`@davidsyoung`), vLLM PR #163](https://github.com/local-inference-lab/vllm/pull/163), incorporated byte-identically into PR #139 with authorship preserved.
- **Runtime foundations:** the [vLLM](https://github.com/vllm-project/vllm), [Local Inference Lab](https://github.com/local-inference-lab), [Gilded Gnosis vLLM](https://github.com/local-inference-lab/vllm), [Sparkinfer](https://github.com/local-inference-lab/sparkinfer), and [ExLlamaV3](https://github.com/turboderp-org/exllamav3) contributors.
- **Validated image publication:** [Verdict AI](https://hub.docker.com/r/verdictai/glm52-exl3-sparkinfer).
- **Benchmark harness:** [Local Inference Lab `llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench), commit `86cf05c2f42f4d21b909b6e684424ca1aab89fd5`, reporting version `0.4.29`.
- **Four-GPU campaign, hardware, experiment direction, measurements, and publication:** [Josh Cartu (`@jcartu`)](https://github.com/jcartu).
- **Disclosed upstream development assistance:** OpenAI Codex and Anthropic Claude Code, as recorded by the authors of PRs #139 and #49; CodeRabbit supplied automated PR review summaries/checks.

See [CREDITS.md](CREDITS.md) for the ownership boundary, source links, and contribution details.

## Breakthrough extension

The follow-on deep dive found and removed a sparse-indexer transient-allocation barrier. The current quality-gated production profile now exposes **999,424 tokens**, up from **786,432 (+27.1%)**, while passing LAVD 10/10, Estonia 10/10, and a fresh OpenAI-compatible tool-call smoke test.

| Result | Validated v20 baseline | Current production |
| --- | ---: | ---: |
| Maximum model length | 786,432 | **999,424** |
| Direct capacity proof | 716,800-token decode | **998,800 prompt + 32 completion, HTTP 200** |
| Cold prefill, 8K | 3,524 tok/s | **3,581 tok/s** |
| Cold prefill, 64K | 2,068 tok/s | **2,098 tok/s** |
| Cold prefill, 128K | 1,937 tok/s | **1,964 tok/s** |
| LAVD | 10/10 | **10/10** |
| Estonia | 10/10 | **10/10** |

The larger KV geometry costs approximately 4–9% short-context decode throughput, depending on concurrency. A native-edge experimental profile executed 1,048,000 prompt tokens plus 32 generated tokens, but missed the Estonia gate at 8/10. Forced execution reached 1,211,800 prompt tokens beyond the model's native limit, but structured retrieval failed; this study makes **no usable-context claim above 1,048,576 tokens**.

The campaign also identified workload-specific speed profiles:

- **DCP1:** up to +20.2% decode throughput at C4, with a 262,144-token ceiling.
- **DCP2 workspace experiment:** up to +54.1% 128K prefill and 373.6 tok/s at C8, with a 524,288-token ceiling, but rejected after LAVD scored 8/10.

See [BREAKTHROUGH_CAMPAIGN.md](BREAKTHROUGH_CAMPAIGN.md) for root cause, full matrices, quality boundaries, rejected routes, upstream research, and next-step candidates. The new byte-for-byte evidence is indexed under [`results/breakthrough/`](results/breakthrough/).

### Issue #34 RC2 follow-on

The 24 July follow-on tested the exact [issue #34 RC2 image](https://github.com/local-inference-lab/rtx6kpro/issues/34) on its supported MXFP8/NVFP4/NF3 hybrid checkpoint path. RC2 does not bundle the EXL3 integration, so this is a measured alternative rather than a drop-in upgrade.

| Result | EXL3 production | Safe RC2 NF3 profile |
| --- | ---: | ---: |
| Topology | TP4/DCP4/MTP3 | TP4/DCP2/MTP3 |
| Maximum model length | **999,424** | 180,224 |
| 64K cold prefill | 2,098 tok/s | **3,340 tok/s (+59.2%)** |
| 128K cold prefill | 1,964 tok/s | **3,168 tok/s (+61.3%)** |
| Zero-context C1 | 98.0 tok/s | **114.5 tok/s (+16.8%)** |
| Zero-context C2 | **153.9 tok/s** | 137.2 tok/s |
| Zero-context C4 | 224.8 tok/s | **227.0 tok/s** |
| Zero-context C8 | 324.7 tok/s | **335.7 tok/s** |
| LAVD exact / near / fail | **6 / 4 / 0** | 1 / 9 / 0 |
| Estonia | 10/10 | 10/10 |

The safe RC2 DCP2 profile also completed a 179,017-token request and an automatic tool-call round trip. The faster DCP4/batch-5,120 candidate failed 3/10 LAVD and was rejected; forced `tool_choice="required"` repeated identical calls until its output cap. The quality-first production selection therefore remains EXL3, while RC2 DCP2 is published as an optional high-prefill/low-latency profile.

See the [RC2 campaign section](BREAKTHROUGH_CAMPAIGN.md#issue-34-rc2-follow-on-24-july-2026), [`results/issue34/`](results/issue34/), its [manifest](results/issue34-manifest.json), and the [publication-safe Compose file](configs/docker-compose.rc2-nf3.yml).

## Original validated v20 baseline

The measurements below preserve the first selected v20 configuration and serve as the comparison baseline for the breakthrough extension.

### Exact-token cold prefill

These are like-for-like client measurements using exact `/tokenize` targeting and one generated token.

| Prompt tokens | v2 baseline | v20 winner | Change | TTFT, baseline → winner |
| ---: | ---: | ---: | ---: | ---: |
| 8,194 | 2,222 tok/s | 3,524 tok/s | **+58.6%** | 3.687 s → 2.325 s |
| 65,538 | 1,438 tok/s | 2,068 tok/s | **+43.8%** | 45.579 s → 31.688 s |
| 131,074 | 1,412 tok/s | 1,937 tok/s | **+37.2%** | 92.825 s → 67.668 s |

Primary artifacts: [`v2 baseline`](results/raw/exl3-v2-baseline-prefill-20260723.json) and [`v20 winner`](results/raw/exl3-v20-winning-prefill-exact-20260723.json).

### Sustained decode at zero input context

Each cell is aggregate output throughput from a 15-second steady-state window after warmup.

| Concurrency | v2 baseline | v20 winner | Change |
| ---: | ---: | ---: | ---: |
| C1 | 58.1 tok/s | 103.3 tok/s | **+77.9%** |
| C2 | 127.2 tok/s | 160.3 tok/s | **+26.0%** |
| C4 | 203.7 tok/s | 239.5 tok/s | **+17.6%** |
| C8 | 298.2 tok/s | 357.5 tok/s | **+19.9%** |

Primary artifacts: [`v2 baseline`](results/raw/exl3-v2-baseline-decode-20260723.json) and [`v20 winning matrix`](results/raw/exl3-v20-winning-decode-matrix-20260723.json).

### Winning decode matrix

| Input context | C1 | C2 | C4 | C8 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 103.3 | 160.3 | 239.5 | 357.5 |
| 32,768 | 103.4 | 161.6 | 234.2 | 346.8 |
| 131,072 | 104.7 | 162.2 | 228.1 | capacity-limited |

Values are aggregate tokens/second. The C8/128K cell could not fit within the configured KV budget and is intentionally omitted rather than presented as a performance result.

## Capacity, quality, and behavior gates

- **716,800-token exact context:** completed a cached-context C1 run at **78.88 aggregate tok/s**, with 1,182 measured output tokens and no request error. The request used a 1,024-token output cap; continuous usage counted the rolling measured window. See the [raw result](results/raw/exl3-v20-mtp3-specdecode-kvrope-700k-exact-decode-20260723.json).
- **900K/1M target in the original geometry:** not safe at that stage. The attempt exceeded the then-configured 786,432-token model/KV budget and encountered an OOM requiring an additional 216 MiB allocation. The follow-on campaign later isolated that transient-allocation barrier; see [BREAKTHROUGH_CAMPAIGN.md](BREAKTHROUGH_CAMPAIGN.md).
- **Estonia long-context diagnostic, default sampling:** **10/10 passed**, zero errors, zero cap hits, 3,576.8 average completion tokens, 59.0 aggregate generation tok/s. See the [final sampled run](results/raw/exl3-v20-final-estonia10-sampled-20260723.json).
- **LAVD, 24,576-token cap:** **10/10 acceptable**: 5 exact, 5 near, 0 fail, zero cap hits, 68.6 aggregate generation tok/s. See the [raw result](results/raw/exl3-v20-winning-lavd10-24k-20260723.json).
- **OpenAI-compatible tool calling:** a fresh smoke request returned HTTP 200, `finish_reason=tool_calls`, and `get_weather({"city":"Paris"})`. See [the narrowed request/response proof](results/tool-call-smoke.json).
- **Thermal/power gate:** the winning decode matrix peaked at 92 °C and 2,084.95 W total GPU power; the exact prefill run peaked at 93 °C and 2,077.79 W; the 716,800-token run peaked at 93 °C and 2,096.59 W. All are below the configured 2,400 W total power limit.

### Important sampling caveat

Forcing `temperature=0` on the Estonia profile caused repetitive generations and output-cap hits on this stack. The deterministic arms scored 4/10 with MTP3/KV-FP8-RoPE enabled, 4/10 with KV-FP8-RoPE disabled, and 0/10 with MTP1/KV-FP8-RoPE disabled. Returning the profile to its default sampling behavior produced 10/10.

That result is reported as a methodology/sampling interaction, not hidden as a failed runtime candidate. The raw deterministic and sampled artifacts are all preserved. LAVD remained a deterministic `temperature=0` gate and passed 10/10 with the 24K output cap.

## Current production serving configuration

- Four-way tensor parallelism and four-way decode-context parallelism (`TP4/DCP4`), DCP A2A.
- Native EXL3 Trellis routed-MoE path with `VLLM_EXL3_PREFILL_BLOCK_M=64` and prefill chunk 3.
- Adaptive sparse-indexer fold with a 256 MiB two-level candidate-buffer cap.
- MTP speculative decoding depth 3 with Triton draft MoE.
- `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1`.
- `KV_FP8_ROPE=1`.
- 3,072 maximum batched tokens and 8 maximum sequences.
- **999,424 maximum model length, 3,904 GPU blocks, and 0.95 GPU memory utilization.**
- Full and piecewise CUDA graphs sized from the selected MTP depth.
- Experimental DCP workspace projection disabled after quality regressions.
- QBMM absorbed BMM disabled: the EXL3 checkpoint does not expose the contiguous ModelOpt MXFP8 `kv_b_proj` layout required by that path.

The exact publication copy is [`configs/docker-compose.exl3-v20.yml`](configs/docker-compose.exl3-v20.yml), which mounts [`configs/runtime-paged-indexer.py`](configs/runtime-paged-indexer.py) and extends [`configs/docker-compose.exl3-experiment.yml`](configs/docker-compose.exl3-experiment.yml).

Pinned final image:

```text
verdictai/glm52-exl3-sparkinfer:v20-gg6722c1d-si1a88b38-cu132-sm120a@sha256:5294b753a81cbed5c7cecd4ef5acdfd1cc13c96bb9233636a42ab8841a439b01
```

The archived winning Compose SHA-256 is recorded in [`results/SHA256SUMS`](results/SHA256SUMS).

## Hardware and software

| Component | Campaign environment |
| --- | --- |
| GPUs | 4 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB each |
| Aggregate VRAM | 391,548 MiB reported |
| Per-GPU power limit | 600 W |
| Driver | 610.43.02 |
| GPU topology | Four GPUs under one NUMA node; GPU-to-GPU path reported as `NODE` |
| Runtime | Gilded Gnosis v20, CUDA 13.2, PyTorch 2.12, Sparkinfer |
| vLLM runtime string | `v0.11.2.dev280+gilded.gnosis.v20.vllm3e731bc.si1a88b38.fi801d57a.cu132.20260722` |
| Checkpoint | `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`, 332.19 GB payload |
| Benchmark | `llm-inference-bench` v0.4.29, commit `86cf05c...` |

The winning decode matrix reported 382,588 MiB average and 382,600 MiB maximum VRAM use, or 97.71% of aggregate reported VRAM.

## What was tested and rejected

The archive preserves all 31 original July 23 JSON artifacts, a curated 16-artifact EXL3 breakthrough extension, and 14 issue #34 RC2 artifacts, including regressions and failed capacity/quality candidates.

- MTP2 had a narrow C8 result comparable to MTP3 but lost materially at C1 and C4.
- MTP4 and MTP5 lost throughput as concurrency increased.
- A 2,048-token scheduler budget regressed C1/C4/C8; both 4,096-token runs were unstable and badly regressed low concurrency.
- The B12X draft-MoE backend did not improve on the Triton control.
- Prefill block sizes 48 and 32 were slower than 64.
- Disabling KV-FP8-RoPE did not resolve deterministic Estonia looping and would give up the selected capacity benefit.
- The QBMM absorbed path was incompatible with this checkpoint layout.
- Quantizing MTP layer 78 was rejected for this campaign: the current encoder excludes it and the runtime lacks the matching Trellis draft path. It requires a separately validated offline rebuild, metadata change, runtime work, and quality/acceptance study.

Detailed numbers and decision rationale are in [METHODOLOGY.md](METHODOLOGY.md).

## Reproduce

### 1. Acquire the upstream checkpoint

Follow the upstream model card and license:

```bash
hf download brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw \
  --local-dir "$HOME/models/GLM-5.2-EXL3-TR3-3.0bpw"
```

The model weights are intentionally not redistributed here.

### 2. Start the pinned service

```bash
export MODEL_DIR="$HOME/models/GLM-5.2-EXL3-TR3-3.0bpw"
# Machine-specific. The campaign host used 3,1,2,0.
export CUDA_VISIBLE_DEVICES="3,1,2,0"

docker compose \
  -f configs/docker-compose.exl3-v20.yml \
  up glm52-exl3-v20
```

The publication copy replaces the campaign host's absolute model path with the required `MODEL_DIR` variable. It otherwise preserves the measured serving geometry.

### 3. Run the benchmark harness

```bash
git clone https://github.com/local-inference-lab/llm-inference-bench.git
cd llm-inference-bench
git checkout 86cf05c2f42f4d21b909b6e684424ca1aab89fd5
```

Representative final commands are documented in [METHODOLOGY.md](METHODOLOGY.md). Exact benchmark arguments, hardware diagnostics, event logs, and per-cell measurements are embedded in each JSON.

## Repository map

```text
configs/                           publication-safe measured Compose/runtime setup
results/raw/                       31 unmodified original campaign JSON artifacts
results/breakthrough/              16 unmodified follow-on campaign artifacts
results/issue34/                 14 curated issue #34 RC2 artifacts
results/manifest.json              original artifact provenance and SHA-256
results/breakthrough-manifest.json follow-on artifact provenance and SHA-256
results/issue34-manifest.json    issue #34 artifact provenance and SHA-256
results/SHA256SUMS                 checksums for all reproducibility artifacts
results/tool-call-smoke.json
METHODOLOGY.md                     original experiment sequence and controls
BREAKTHROUGH_CAMPAIGN.md           follow-on root cause, profiles, gates, and limits
CREDITS.md                         ownership and attribution ledger
```

See [results/README.md](results/README.md) for the artifact index and verification instructions.

## Integrity and scope

- The 31 files in `results/raw/`, 16 files in `results/breakthrough/`, and 14 files in `results/issue34/` preserve the campaign evidence; benchmark outputs are byte-for-byte copies, while the direct-observation JSON records API and startup checks not emitted by the harness.
- Generated manifests, runtime/configuration copies, and documentation are checksummed.
- Hostname, driver, PCI topology, hardware samples, and benchmark event logs remain in the raw evidence because they affect reproducibility.
- No model weights, container layers, credentials, or API keys are included; the only modified upstream source copies are the two runtime files explicitly listed and attributed in [CREDITS.md](CREDITS.md).
- Results apply to the pinned checkpoint, image digest, four-GPU topology, and stated benchmark methodology. They should not be generalized to other quantizations, GPU counts, runtimes, or sampling policies without measurement.

## License

Original study documentation, public-safe configuration copies, and generated result metadata are released under the [MIT License](LICENSE). Referenced models, runtimes, libraries, container images, and their source code retain their own licenses and copyright notices. See [CREDITS.md](CREDITS.md) before redistributing any upstream component.
