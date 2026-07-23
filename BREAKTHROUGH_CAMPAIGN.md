# Performance and Context Breakthrough Campaign

This report documents the follow-on 23 July 2026 campaign that continued beyond the repository's original validated v20 result. The objective was to search broadly for higher throughput and more usable context, preserve failed routes, and select a production profile only after capacity, quality, and tool-call gates.

The work was performed on the same four RTX PRO 6000 Blackwell Workstation Edition GPUs, pinned 3.0 bpw rank-sliced EXL3 checkpoint, immutable Verdict AI v20 image digest, and `llm-inference-bench` v0.4.29 environment described in [METHODOLOGY.md](METHODOLOGY.md).

## Executive result

The campaign found a real long-context memory failure in Sparkinfer's paged sparse-indexer fold and replaced its binary long-context switch with an adaptive memory cap. The selected production profile now exposes **999,424 tokens**, up from **786,432 tokens (+27.1%)**, while preserving **10/10 LAVD**, **10/10 Estonia**, and a fresh OpenAI-compatible tool-call pass.

A direct request with **998,800 prompt tokens plus 32 generated tokens** returned HTTP 200. Short-context cold prefill also improved by 1.4–1.6% over the already optimized v20 baseline. The cost of the larger production geometry is 4–9% lower short-context decode throughput, depending on concurrency.

The model also executed a **1,048,000-token prompt plus 32 generated tokens** at its native 1,048,576-token position limit. That native-edge profile passed LAVD 10/10 but only Estonia 8/10, so it is documented as an experimental capacity proof rather than the production default.

Physical allocation was pushed to a 1,211,800-token prompt, but structured retrieval failed beyond the model's native positional limit. This study therefore makes **no usable-context claim above 1,048,576 tokens**.

## Selected production profile

| Control | Selected value |
| --- | --- |
| Parallelism | TP4 / DCP4, A2A |
| MTP | Depth 3, Triton draft MoE |
| Maximum model length | **999,424** |
| GPU blocks | **3,904** |
| Maximum batched tokens | 3,072 |
| Maximum sequences | 8 |
| GPU memory utilization | 0.95 |
| KV format | `nvfp4_ds_mla`, `KV_FP8_ROPE=1` |
| EXL3 prefill | Trellis M64, chunk 3 |
| Sparse-indexer fold | Adaptive, 256 MiB candidate-buffer cap |
| Experimental DCP workspace projection | Disabled |
| Image | `verdictai/glm52-exl3-sparkinfer:v20-gg6722c1d-si1a88b38-cu132-sm120a@sha256:5294b753a81cbed5c7cecd4ef5acdfd1cc13c96bb9233636a42ab8841a439b01` |

Publication files:

- [`configs/docker-compose.exl3-v20.yml`](configs/docker-compose.exl3-v20.yml)
- [`configs/runtime-paged-indexer.py`](configs/runtime-paged-indexer.py)
- [`results/breakthrough/exl3-breakthrough-direct-evidence-20260723.json`](results/breakthrough/exl3-breakthrough-direct-evidence-20260723.json)

## Production performance

### Exact-token cold prefill

The baseline in this table is the original validated v20 winner, not the much slower v2 image.

| Prompt tokens | Validated v20 baseline | Final 999,424 profile | Change |
| ---: | ---: | ---: | ---: |
| 8,194 | 3,524 tok/s | **3,581 tok/s** | **+1.6%** |
| 65,538 | 2,068 tok/s | **2,098 tok/s** | **+1.5%** |
| 131,074 | 1,937 tok/s | **1,964 tok/s** | **+1.4%** |

Evidence: [`production prefill`](results/breakthrough/exl3-breakthrough-production-b3072-prefill-20260723.json).

### Sustained decode matrix

Each value is aggregate output tokens/second from the same duration-mode harness used in the original study.

| Input context | C1 | C2 | C4 | C8 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 98.0 | 153.9 | 224.8 | 324.7 |
| 32,768 | 97.8 | 146.4 | 208.5 | 318.6 |
| 131,072 | 96.3 | 151.4 | 209.1 | capacity-limited |

Relative to the 786,432-token v20 baseline at zero context, these cells trade approximately 5.1%, 4.0%, 6.1%, and 9.2% at C1/C2/C4/C8 for the 27.1% larger logical context budget.

Evidence: [`production decode`](results/breakthrough/exl3-breakthrough-production-b3072-decode-20260723.json).

### Production gates

| Gate | Result |
| --- | --- |
| Direct capacity | 998,800 prompt + 32 completion tokens, HTTP 200 |
| LAVD | **10/10 acceptable**: 6 exact, 4 near, 0 fail |
| Estonia | **10/10 pass**, 0 fail |
| Tool calling | HTTP 200, `finish_reason=tool_calls`, `get_weather({"city":"Tallinn"})` |

Evidence: [`direct requests`](results/breakthrough/exl3-breakthrough-direct-evidence-20260723.json), [`LAVD`](results/breakthrough/exl3-breakthrough-b3072-quality-lavd10-20260723.json), and [`Estonia`](results/breakthrough/exl3-breakthrough-b3072-quality-estonia10-20260723.json).

## Breakthrough 1: bounded adaptive sparse-indexer folding

### Root cause

At long context, `sparkinfer/attention/nsa_indexer/paged.py` used a fast two-level fold that materialized float32 values and int32 indices with shape approximately:

```text
(q_rows × total_slices, topk)
```

At the 786K boundary, one `fold_indices` request alone reached roughly 312 MiB. That transient allocation, rather than the persistent KV cache itself, caused the observed long-context OOM.

A streaming carry fold already existed in the runtime and used bounded persistent scratch, but selecting it required globally disabling the faster two-level route. The local change makes selection adaptive:

```text
candidate_bytes = q_rows × total_slices × topk × 8 + q_rows × 4
```

- If the combined candidate buffers are at or below `SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB`, the fast two-level route is retained.
- If the estimate exceeds the cap, the operation falls back to the existing streaming carry fold.
- `SPARKINFER_INDEXER_TWO_LEVEL_FOLD=0|1|auto` remains available for explicit controls.

The production cap is 256 MiB. This is the primary enabler for the move from 786,432 to 999,424 tokens without changing checkpoint quantization.

### Why this is safer than a blanket disable

The branch is based on the exact candidate shape that would be allocated. It therefore preserves the faster implementation for ordinary context lengths and only selects the bounded path where the transient allocation is actually dangerous. The fallback is existing runtime logic, not a new approximate computation.

## Breakthrough 2: specialized DCP performance profiles

The campaign found that one serving geometry is not optimal for every workload.

### DCP1: decode-speed profile

At zero input context, DCP1 measured:

| Concurrency | DCP4 baseline | DCP1 | Change |
| ---: | ---: | ---: | ---: |
| C1 | 103.3 | **116.1** | **+12.4%** |
| C2 | 160.3 | **191.7** | **+19.6%** |
| C4 | 239.5 | **287.9** | **+20.2%** |
| C8 | 357.5 | **371.5** | **+3.9%** |

The tradeoff is a 262,144-token capacity ceiling. This profile is not promoted to a default because the safe DCP1 query-BMM fix tracked in [local-inference-lab/vllm PR #173](https://github.com/local-inference-lab/vllm/pull/173) was not available from the referenced container tag during the campaign.

Evidence: [`DCP1 decode`](results/breakthrough/exl3-breakthrough-dcp1-decode-20260723.json).

### DCP2: high-prefill experimental profile

The patched DCP2 workspace topology produced the campaign's largest measured speed gains:

| Prompt tokens | DCP4 baseline | DCP2 workspace | Change |
| ---: | ---: | ---: | ---: |
| 8,194 | 3,524 | **3,907** | **+10.9%** |
| 65,538 | 2,068 | **3,096** | **+49.7%** |
| 131,074 | 1,937 | **2,984** | **+54.1%** |

Its zero-context decode matrix was 107.4 / 171.6 / 245.7 / 373.6 tok/s at C1/C2/C4/C8, with a 524,288-token capacity ceiling. A 520,000-token structured-retrieval request returned the expected code after reusing a prefix computed by the immediately preceding 524,100-token request; it is evidence of near-edge retrieval, not a cold-prefill latency measurement. Estonia passed 10/10.

However, deterministic LAVD scored only 8/10. The profile is therefore **experimental and rejected for production**, despite its strong performance.

Evidence: [`prefill`](results/breakthrough/exl3-breakthrough-dcp2-workspace-prefill-20260723.json), [`decode`](results/breakthrough/exl3-breakthrough-dcp2-workspace-decode-matrix-20260723.json), [`LAVD`](results/breakthrough/exl3-breakthrough-dcp2-workspace-lavd10-20260723.json), and [`Estonia`](results/breakthrough/exl3-breakthrough-dcp2-workspace-estonia10-20260723.json).

### MTP2 high-concurrency variant

DCP2 with MTP2 reached 381.9 tok/s at C8, 2.2% above DCP2/MTP3, but C1 fell to 94.1 tok/s, 12.4% below DCP2/MTP3. MTP3 remains the balanced choice.

Evidence: [`DCP2 MTP2`](results/breakthrough/exl3-breakthrough-dcp2-mtp2-decode-20260723.json).

## Native-edge and beyond-native limits

### Native-edge capacity

With DCP4, 4,096 blocks, batch 2,816, adaptive folding, and the experimental workspace projection disabled:

- 1,048,000 prompt tokens plus 32 completion tokens returned HTTP 200.
- LAVD passed 10/10.
- Estonia scored 8/10.
- Exact 716,800-token cached-context decode measured 86.70 aggregate tok/s, versus 78.88 tok/s in the original baseline (+9.9%). The harness labeled the cell capacity-limited because its warmup timed out after 377.9 seconds, although the request itself generated 2,600 tokens with zero errors; the raw artifact is retained with that caveat.

Because the Estonia gate did not pass, 1,048,576 is a proven execution boundary, not the selected production default.

Evidence: [`direct capacity`](results/breakthrough/exl3-breakthrough-direct-evidence-20260723.json), [`LAVD`](results/breakthrough/exl3-breakthrough-final-no-workspace-lavd10-20260723.json), [`Estonia`](results/breakthrough/exl3-breakthrough-final-safe-estonia10-20260723.json), and [`716,800 decode`](results/breakthrough/exl3-breakthrough-final-dcp4-exact716800-decode-20260723.json).

### Beyond-native failure boundary

The checkpoint declares `max_position_embeddings=1048576` with no RoPE scaling. Forced configurations physically executed prompts at 1,179,000 and 1,211,800 tokens, but a 1,100,011-token structured-retrieval request returned `79979979974V` instead of `CERULEAN-ORBIT-7391`.

This is the critical distinction:

- **Memory capacity:** at least 1.21M tokens can be allocated and executed.
- **Validated usable context:** no claim above the native 1,048,576-token positional limit.
- **Selected quality-gated production context:** 999,424 tokens.

## Workspace projection: performance win, quality rejection

The v20 source contains latent controls that project DCP query state before merge and place the gather in a persistent workspace:

- `VLLM_DCP_PROJECT_BEFORE_MERGE=1`
- `VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=1`

The local experimental source fixed a final-partial-chunk output-stride bug and added the `(TP4, DCP2)` topology. This removed a large transient gather and materially improved prefill.

It was not production-safe:

- DCP4 workspace variants scored 7/10 and 8/10 on LAVD.
- Disabling the workspace path restored DCP4 LAVD to 10/10.
- DCP2 workspace scored 8/10 on LAVD.

The source and overlay are retained for upstream debugging, but the production Compose file mounts only the adaptive paged-indexer source. See [`configs/runtime-b12x-mla.py`](configs/runtime-b12x-mla.py) and [`configs/docker-compose.exl3-fold-experiment.yml`](configs/docker-compose.exl3-fold-experiment.yml).

## Other routes tested

| Route | Observation | Decision |
| --- | --- | --- |
| Trellis prefill M48/M32 | Reduced arena memory from 1,054.2 MiB to 1,030.6/1,007.1 MiB but did not make batch 3,072 / 4,096 blocks fit | Keep M64 |
| Prefill chunk 128 → 3 | Saved approximately 77.4 MiB/GPU; MTP3 requires a minimum M of 4, not a 128-token chunk | Adopt chunk 3 |
| MTP depth 2/3/4/5 | MTP3 remained best balanced; MTP2 only won the DCP2 C8 niche | Keep MTP3 |
| B12X vs Triton draft MoE | No significant B12X gain | Keep Triton |
| Batch 2,048/4,096 | Regressed or unstable | Keep 3,072 production / 2,816 native-edge |
| Selected CKV decode prefetch | Regressed DCP C4 by roughly 5–6% | Reject |
| DCP2 without workspace at 4,096 blocks | 108 MiB allocation with only 97.8 MiB free | Does not fit |
| QBMM absorbed BMM | Checkpoint lacks the required contiguous ModelOpt MXFP8 `kv_b_proj` layout | Reject for this checkpoint |
| Beyond-native RoPE override | Physical execution succeeded; retrieval failed | Never deploy |

## Online deep-dive findings

The experiment matrix was informed by current upstream implementation and optimization work:

- [vLLM performance optimization](https://docs.vllm.ai/en/stable/configuration/optimization/) and [memory conservation](https://docs.vllm.ai/en/latest/configuration/conserving_memory/) document the scheduler, graph, compilation, and memory controls used to structure the sweeps.
- [vLLM cache configuration](https://docs.vllm.ai/en/stable/api/vllm/config/cache/) defines the explicit KV-cache controls and override boundary.
- The [EXL3 checkpoint model card](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw), [EXL3 backend PR #139](https://github.com/local-inference-lab/vllm/pull/139), [planned-prefill PR #163](https://github.com/local-inference-lab/vllm/pull/163), and [Sparkinfer PR #49](https://github.com/local-inference-lab/sparkinfer/pull/49) establish the checkpoint/runtime path.
- [MTP verifier PR #164](https://github.com/local-inference-lab/vllm/pull/164) is already present in the production image.
- [CKV prefetch PR #160](https://github.com/local-inference-lab/vllm/pull/160), its [decode-regression follow-up PR #161](https://github.com/local-inference-lab/vllm/pull/161), [high-utilization profiling PR #172](https://github.com/local-inference-lab/vllm/pull/172), and [safe DCP1 query-BMM PR #173](https://github.com/local-inference-lab/vllm/pull/173) were evaluated against the measured bottlenecks.
- [MXFP8 memory PR #154](https://github.com/local-inference-lab/vllm/pull/154) does not apply to this EXL3 checkpoint layout.
- Quantized-draft research, including [QuantSpec](https://arxiv.org/html/2505.22179v1), [Quantized Speculative Decoding](https://arxiv.org/html/2503.13565v1), and [Judge Decoding](https://arxiv.org/html/2410.11305v2), supports a future offline-quantized MTP investigation rather than an unvalidated live patch.

## Highest-probability next breakthroughs

1. **Upstream the adaptive fold.** Convert the local source mount into a reviewed Sparkinfer change with allocation-boundary and equivalence tests.
2. **Diagnose workspace numerical divergence.** The speedup is too large to ignore, but the LAVD regressions prohibit deployment. Compare projected-query and merged-query outputs across full and final partial chunks before attempting another performance run.
3. **Re-test DCP1 with the safe query-BMM image.** DCP1 is the clearest decode-speed profile once PR #173 is available in an immutable image.
4. **Quantize MTP layer 78 offline.** The draft layer is wholly BF16: 791 tensors and approximately 18.54 GiB checkpoint-wide, or 4.635 GiB/GPU at TP4. A 3 bpw representation could theoretically recover roughly 3.77 GiB/GPU, but the current encoder explicitly excludes layer 78 and the runtime lacks the matching validated Trellis draft path. This requires a new checkpoint artifact, metadata, runtime support, and a full acceptance/quality study.

## Reproduce the selected service

From the repository root:

```bash
export MODEL_DIR="$HOME/models/GLM-5.2-EXL3-TR3-3.0bpw"
export CUDA_VISIBLE_DEVICES="3,1,2,0"  # campaign host mapping; machine-specific

docker compose \
  -f configs/docker-compose.exl3-v20.yml \
  up glm52-exl3-v20
```

The Compose file extends the publication-safe experiment base in the same directory and mounts the adaptive paged-indexer source. The experimental workspace overlay is intentionally not part of this command.

## Claim boundary

- Measurements apply only to the pinned checkpoint, immutable image digest, four-GPU PCIe topology, power limits, and harness methodology.
- A successful allocation or HTTP response is not treated as usable long context without a structured-retrieval or quality gate.
- DCP1 and DCP2 results are workload-specific profiles, not universal replacements for DCP4.
- Experimental workspace source is published for reproducibility and debugging, not as a production recommendation.
- No result above the model's 1,048,576-token native positional limit is presented as correct or deployable.
