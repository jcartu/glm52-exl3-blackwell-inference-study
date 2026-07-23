# Methodology and Experiment Log

This document records the complete 23 July 2026 GLM-5.2 EXL3/Trellis tuning process, including controls, candidate configurations, regressions, capacity limits, and final gates.

> Ownership notice: the campaign measured and configured upstream work; it did not create or take ownership of GLM-5.2, the EXL3 checkpoint, vLLM, Sparkinfer, ExLlamaV3/Trellis, Gilded Gnosis, or the referenced container images. See [CREDITS.md](CREDITS.md) for component-level attribution.

## 1. Question

For `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` on four RTX PRO 6000 Blackwell 96 GB GPUs, determine the highest-performing configuration that:

1. preserves the checkpoint's native rank-sliced EXL3/Trellis routed experts;
2. improves both cold prefill and sustained decode rather than optimizing one isolated cell;
3. remains stable through C8 and long input contexts;
4. preserves a practical long-context KV budget;
5. passes long-context reasoning, tool-call, power, and thermal gates; and
6. is pinned to an immutable, publicly retrievable runtime image.

## 2. Ownership and upstream starting point

The campaign began from work already completed upstream:

- Z.ai's GLM-5.2 base model.
- Brandon Music's TP4, rank-sliced 3.0 bpw EXL3 checkpoint, vLLM EXL3 loading/execution integration, and Sparkinfer planned Trellis fused-MoE path.
- David Young's dual-plan Trellis prefill dispatch, later incorporated byte-identically into Brandon's vLLM PR #139 with authorship preserved.
- The Gilded Gnosis v20 runtime and its MTP/DCP correctness work.
- Local Inference Lab's `llm-inference-bench` harness.

The work in this repository is the campaign design, serving-parameter sweep, evidence capture, configuration selection, validation, and publication. Full links and licenses are in [CREDITS.md](CREDITS.md).

## 3. Controlled environment

### Hardware

- 4 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB.
- 391,548 MiB aggregate VRAM reported by the benchmark harness.
- 600 W configured limit per GPU, 2,400 W aggregate.
- NVIDIA driver 610.43.02.
- All GPUs under NUMA node 0; `nvidia-smi topo -m` reported `NODE` between GPU pairs.
- NVIDIA P2P override runtime values matched the host's configured expected values. `p2pmark` was not run, so this campaign does not substitute the override check for an independent P2P bandwidth claim.

### Final runtime and checkpoint

```text
checkpoint: brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw
image: verdictai/glm52-exl3-sparkinfer:v20-gg6722c1d-si1a88b38-cu132-sm120a
image digest: sha256:5294b753a81cbed5c7cecd4ef5acdfd1cc13c96bb9233636a42ab8841a439b01
Gilded Gnosis vLLM: 6722c1d
Sparkinfer: 1a88b389a8d14f26dbe4c157965938cfd8f1bf51
runtime string: v0.11.2.dev280+gilded.gnosis.v20.vllm3e731bc.si1a88b38.fi801d57a.cu132.20260722
```

### Benchmark harness

- Repository: <https://github.com/local-inference-lab/llm-inference-bench>
- Commit: `86cf05c2f42f4d21b909b6e684424ca1aab89fd5`
- Version embedded in results: `0.4.29`
- Server: local OpenAI-compatible endpoint on port 5001.

Each raw JSON embeds the exact CLI argument projection, runtime discovery, hardware diagnostics, event log, methodology definitions, and measurements. These embedded fields are the authority where a descriptive summary and an artifact differ.

## 4. Measurement definitions

### Sustained decode

The primary decode signal is aggregate completion tokens per measured steady-state window, using vLLM's continuous OpenAI stream usage where available. Each reported primary matrix cell ran for 15 seconds after a 3-second warmup. `ignore_eos=true` keeps request streams active for the duration window.

Aggregate throughput is intentional: the study asks how much useful output the four-GPU service produces at each concurrency. Per-request rates remain in the raw JSON.

### Cold prefill

Primary prefill comparisons use exact tokenizer targeting:

- requested contexts: 8K, 64K, and 128K;
- actual prompt tokens: 8,194, 65,538, and 131,074;
- client metric: `prompt_tokens / TTFT`;
- one output token;
- four cold samples per context in the final exact run.

The archive also contains estimate-targeted prefill runs. They are retained for process completeness but are not used for the headline A/B because prompt-token counts are not the same apples-to-apples control.

### Capacity

The 700K test used exact tokenizer targeting at 716,800 input tokens, C1, a 15-second measurement, a 1,024-token output cap, a 900-second warmup timeout, and a 786,432-token maximum total/model budget.

### Quality profiles

- **Estonia:** 707,372-character long-context dense-MLA versus sparse-attention diagnostic. Correctness requires identifying Estonia.
- **LAVD:** long structured-context consistency and ledger correction task. Expected final values are 72 tickets and 46 hours; the harness scores exact, near, or fail.
- Both final quality gates used 10 runs at fixed concurrency 5 and no prefill scout.

## 5. Experiment sequence

### Phase A: establish the v2 baseline

The v2 baseline used the validated EXL3 runtime before the planned prefill improvement and v20 rebase.

| Metric | Baseline |
| --- | ---: |
| Prefill 8,194 tokens | 2,222 tok/s, 3.687 s TTFT |
| Prefill 65,538 tokens | 1,438 tok/s, 45.579 s TTFT |
| Prefill 131,074 tokens | 1,412 tok/s, 92.825 s TTFT |
| Decode C1 / C2 / C4 / C8, context 0 | 58.1 / 127.2 / 203.7 / 298.2 tok/s |

Artifacts:

- [`exl3-v2-baseline-prefill-20260723.json`](results/raw/exl3-v2-baseline-prefill-20260723.json)
- [`exl3-v2-baseline-decode-20260723.json`](results/raw/exl3-v2-baseline-decode-20260723.json)

### Phase B: route prefill through a second planned Trellis path

David Young's vLLM PR #163 added a second Trellis plan sized for prefill batches while retaining the existing decode plan. The historical A/B overlay in [`configs/Dockerfile.exl3-v2-prefill`](configs/Dockerfile.exl3-v2-prefill) copied that isolated `exl3.py` implementation over the pinned v2 image.

Prefill block-size candidates 64, 48, and 32 were measured. Block 64 was retained. The 48/32 raw files are preserved:

- [`block 48`](results/raw/exl3-v2-trellis-m2-block48-prefill-20260723.json)
- [`block 32`](results/raw/exl3-v2-trellis-m2-block32-prefill-20260723.json)
- [`selected planned-prefill run`](results/raw/exl3-v2-trellis-m2-prefill-20260723.json)

The campaign did not claim authorship of this implementation. PR #163 documents the design, measured upstream gains, and the byte-identical incorporation into PR #139.

### Phase C: rebase onto the pinned v20 image

The selected final image rebased EXL3/Trellis on the Gilded Gnosis v20 canonical vLLM/Sparkinfer heads and included the upstream MTP/DCP correctness fixes described by the checkpoint model card and PR #139.

This phase verified that the service loaded all shards, initialized EXL3/Trellis, captured the configured graph sizes, and exposed a healthy OpenAI-compatible endpoint before tuning resumed.

### Phase D: sweep MTP speculative depth

With speculative extension enabled, zero-context C1/C4/C8 were measured for MTP depths 2 through 5.

| MTP depth | C1 | C4 | C8 | Decision |
| ---: | ---: | ---: | ---: | --- |
| 2 | 96.2 | 234.6 | 357.5 | C8 competitive, lower C1/C4 |
| 3 | **103.6** | **243.6** | 352.4 | selected balanced depth |
| 4 | 101.5 | 235.4 | 283.4 | high-concurrency regression |
| 5 | 101.0 | 229.0 | 255.3 | larger high-concurrency regression |

All values are aggregate tok/s. MTP3 was selected because it maximized the balanced service objective, not because it won every isolated cell. A later final matrix measured 357.5 tok/s at C8, showing normal run-to-run variation around the earlier 352.4 value.

Artifacts: `exl3-v20-mtp{2,3,4,5}-specdecode-decode-20260723.json` in [`results/raw`](results/raw/).

### Phase E: enable speculative MLA extension as decode

`VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1` was retained. It materially improved MTP decode in this runtime and was already enabled for downstream candidate comparisons. Later numbers must therefore not claim its gain a second time.

The archive preserves both pre-spec-extension and post-spec-extension MTP3 runs:

- [`MTP3 before the retained extension configuration`](results/raw/exl3-v20-mtp3-decode-20260723.json)
- [`MTP3 with speculative extension`](results/raw/exl3-v20-mtp3-specdecode-decode-20260723.json)

### Phase F: use KV-FP8 RoPE to extend capacity

`KV_FP8_ROPE=1` was approximately throughput-neutral in the measured decode cells but allowed the selected larger KV geometry. The service was raised to:

- `--max-model-len 786432`
- `--num-gpu-blocks-override 3072`
- 3,072 blocks × 64 tokens × DCP4 = 786,432 logical KV tokens.

The exact 716,800-token run completed at 78.8837 aggregate tok/s with no errors. The hardware monitor recorded 382,437 MiB average VRAM use, 382,476 MiB maximum, 1,576.62 W average total GPU power, 2,096.59 W maximum, and 93 °C maximum temperature.

A 900K/1M-direction attempt was retained as a failed boundary probe. It exceeded the configured 786,432-token model/KV geometry and encountered OOM on an additional 216 MiB allocation. Therefore:

- **716,800 exact input tokens are demonstrated.**
- **786,432 is the configured logical maximum, not a demonstrated full-length generation claim.**
- **900K and 1M are not safe or claimed.**

Artifacts:

- [`exact 716,800`](results/raw/exl3-v20-mtp3-specdecode-kvrope-700k-exact-decode-20260723.json)
- [`900K boundary attempt`](results/raw/exl3-v20-mtp3-specdecode-kvrope-900k-decode-20260723.json)
- two earlier 700K calibration/retry artifacts in [`results/raw`](results/raw/)

### Phase G: sweep scheduler batch-token budget

| Maximum batched tokens | C1 | C4 | C8 | Decision |
| ---: | ---: | ---: | ---: | --- |
| 2,048 | 56.8 | 215.2 | 299.3 | rejected |
| 3,072 | approximately 103 | approximately 240–244 | approximately 352–362 | selected |
| 4,096, run 1 | 19.4 | 61.7 | 328.5 | rejected; severe low-C regression |
| 4,096, retry | 45.0 | 113.6 | 318.1 | rejected; regression reproduced |

The 3,072-token setting was retained. The 4,096-token arm was not selected based on one bad run; a retry reproduced the low-concurrency failure mode.

### Phase H: compare draft MoE backend

Matched zero-context A/B:

| Draft backend | C1 | C4 | C8 |
| --- | ---: | ---: | ---: |
| Triton control | **103.1** | **243.6** | **362.3** |
| B12X | 101.4 | 242.0 | 362.1 |

B12X provided no measurable service-level advantage here. Triton remained the selected draft backend.

Artifacts:

- [`Triton control`](results/raw/exl3-v20-mtp3-specdecode-kvrope-drafttriton-control-decode-20260723.json)
- [`B12X candidate`](results/raw/exl3-v20-mtp3-specdecode-kvrope-draftb12x-decode-20260723.json)

### Phase I: audit, but do not force, incompatible paths

#### QBMM absorbed BMM

The runtime's absorbed-BMM optimization requires a contiguous ModelOpt MXFP8 `kv_b_proj` representation. This checkpoint stores the applicable routed weights in rank-sliced EXL3 form and falls back to materialized absorbed weights. `VLLM_B12X_ABSORB_BMM=0` is therefore explicit in the winning overlay.

This is a compatibility conclusion, not a claim that the upstream optimization is generally ineffective.

#### Quantizing MTP layer 78

The checkpoint model card states that MTP layer 78 remains BF16. Quantizing it was rejected in this campaign because the current encoder excludes that layer and the serving runtime does not implement a matching Trellis MTP draft path. A safe experiment would require:

1. an offline checkpoint rebuild;
2. metadata/schema changes;
3. runtime loader and draft-path support;
4. acceptance-rate, quality, graph-capture, and long-context validation.

The theoretical memory recovery was estimated at roughly 3.51 GB/GPU, but no unvalidated checkpoint mutation was shipped to chase that estimate.

### Phase J: final performance matrix

Final zero/32K/128K matrix:

| Context | C1 | C2 | C4 | C8 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 103.3028 | 160.3311 | 239.5431 | 357.5296 |
| 32,768 | 103.3961 | 161.6390 | 234.1725 | 346.7605 |
| 131,072 | 104.7404 | 162.2028 | 228.1219 | capacity-limited |

The run recorded:

- 382,588 MiB average and 382,600 MiB maximum aggregate VRAM use (97.71%);
- 1,464.26 W average and 2,084.95 W maximum aggregate GPU power;
- 92 °C maximum GPU temperature;
- no reported request error in the 11 completed cells.

Artifact: [`exl3-v20-winning-decode-matrix-20260723.json`](results/raw/exl3-v20-winning-decode-matrix-20260723.json).

### Phase K: final exact-token prefill

| Prompt tokens | Baseline tok/s | Winner tok/s | Change | Baseline TTFT | Winner TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,194 | 2,222 | 3,524 | +58.6% | 3.687 s | 2.325 s |
| 65,538 | 1,438 | 2,068 | +43.8% | 45.579 s | 31.688 s |
| 131,074 | 1,412 | 1,937 | +37.2% | 92.825 s | 67.668 s |

The final exact run recorded 1,577.18 W average and 2,077.79 W maximum aggregate GPU power, 93 °C maximum temperature, and 382,600 MiB VRAM use.

### Phase L: quality and behavior gates

#### LAVD

An initial 16,384-token cap produced 9/10 acceptable results with one cap hit. Raising only the output cap to 24,576 produced:

- 10/10 acceptable;
- 5 exact, 5 near, 0 fail;
- 0 cap hits;
- 13,666.1 average completion tokens;
- 68.6326 aggregate generation tok/s;
- 0.794 s average TTFT.

This shows why the campaign did not misclassify a cap-limited answer as a model/runtime quality failure.

Artifacts:

- [`16K cap`](results/raw/exl3-v20-winning-lavd10-16k-20260723.json)
- [`24K cap`](results/raw/exl3-v20-winning-lavd10-24k-20260723.json)

#### Estonia and sampling-policy interaction

Three deterministic `temperature=0` arms were retained:

| Runtime arm | Passes | Cap hits | Observation |
| --- | ---: | ---: | --- |
| MTP3 + KV-FP8-RoPE | 4/10 | 5 | repetitive/cap-limited failures |
| MTP3, KV-FP8-RoPE off | 4/10 | 4 | disabling KV path did not fix it |
| MTP1, KV-FP8-RoPE off | 0/10 | 10 | all runs reached 40K cap |

The profile's default sampling policy then produced:

- 10/10 pass;
- 0 errors and 0 cap hits;
- 3,576.8 average completion tokens, p50 3,132, p90 6,488.9;
- 59.0 aggregate generation tok/s;
- 35.04 s average TTFT, reflecting two prefix-cache waves in the fixed C5 run.

The conclusion is narrow: greedy sampling is not an appropriate Estonia quality gate for this configuration. It is not evidence that KV-FP8-RoPE or MTP3 independently causes the failure. LAVD remained deterministic and passed.

Artifacts:

- [`greedy MTP3/KV on`](results/raw/exl3-v20-winning-estonia10-20260723.json)
- [`greedy MTP3/KV off`](results/raw/exl3-v20-kvrope0-estonia10-20260723.json)
- [`greedy MTP1/KV off`](results/raw/exl3-v20-mtp1-estonia10-20260723.json)
- [`default-sampled final`](results/raw/exl3-v20-final-estonia10-sampled-20260723.json)

#### Tool-call smoke

A fresh OpenAI-compatible request declared a `get_weather` function and asked the model to use it for Paris. The final service returned:

```text
HTTP 200
finish_reason: tool_calls
function: get_weather
arguments: {"city": "Paris"}
```

The narrowed request and observed contract are stored in [`results/tool-call-smoke.json`](results/tool-call-smoke.json). The artifact intentionally excludes unrelated response fields.

## 6. Final serving command

The final command is defined in [`configs/docker-compose.exl3-v20.yml`](configs/docker-compose.exl3-v20.yml). Its effective material settings are:

```text
TP4 / DCP4 A2A
EXL3 Trellis decode plan + block-64 prefill plan
MTP3, greedy draft sampling, Triton draft MoE
VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1
KV_FP8_ROPE=1
max_num_batched_tokens=3072
max_num_seqs=8
max_model_len=786432
num_gpu_blocks_override=3072
gpu_memory_utilization=0.93
async scheduling disabled
```

The graph capture sizes are generated from MTP depth. For MTP3 they are `[4,8,12,16,20,24,28,32]` with a maximum capture size of 32 and Trellis minimum M of 4.

## 7. Representative benchmark commands

Run from the pinned `llm-inference-bench` checkout.

### Final decode matrix

```bash
python llm_decode_bench.py \
  --host localhost --port 5001 --model GLM-5.2 \
  --concurrency 1,2,4,8 --contexts 0,32k,128k \
  --duration 15 --skip-prefill --temperature 0 \
  --max-tokens 8192 --max-total-tokens 786432 \
  --display-mode plain \
  --output exl3-v20-winning-decode-matrix-20260723.json
```

### Exact-token prefill

```bash
python llm_decode_bench.py \
  --host localhost --port 5001 --model GLM-5.2 \
  --prefill-only --prefill-contexts 8k,64k,128k \
  --prefill-duration 10 --prefill-metric client \
  --token-targeting exact --temperature 0 --max-tokens 1 \
  --display-mode plain \
  --output exl3-v20-winning-prefill-exact-20260723.json
```

### Exact 716,800-token context

```bash
python llm_decode_bench.py \
  --host localhost --port 5001 --model GLM-5.2 \
  --concurrency 1 --contexts 700k --duration 15 \
  --skip-prefill --token-targeting exact --temperature 0 \
  --max-tokens 1024 --max-total-tokens 786432 \
  --cell-warmup-timeout-seconds 900 \
  --display-mode plain \
  --output exl3-v20-mtp3-specdecode-kvrope-700k-exact-decode-20260723.json
```

### LAVD final

```bash
python llm_decode_bench.py \
  --host localhost --port 5001 --model GLM-5.2 \
  --test-profile lavd-test --completion-stats-runs 10 \
  --profile-concurrency 5 --completion-stats-temperature 0 \
  --max-tokens 24576 --completion-stats-no-prefill-scout \
  --display-mode plain \
  --output exl3-v20-winning-lavd10-24k-20260723.json
```

### Estonia final

```bash
python llm_decode_bench.py \
  --host localhost --port 5001 --model GLM-5.2 \
  --test-profile estonia --completion-stats-runs 10 \
  --profile-concurrency 5 --max-tokens 40000 \
  --completion-stats-no-prefill-scout --display-mode plain \
  --output exl3-v20-final-estonia10-sampled-20260723.json
```

No explicit completion-stats temperature is supplied in the final Estonia command; this is deliberate and is part of the reported sampling-policy conclusion.

## 8. Interpretation boundaries

- This is one four-GPU Blackwell workstation topology, one EXL3 checkpoint, and one pinned runtime image.
- Aggregate tokens/second is not single-user latency. Both are available in the raw artifacts.
- A configured context budget is not the same as a completed generation at that exact maximum.
- The 716,800-token test used a cached calibrated prefix; it validates capacity and sustained decode after context ingestion, not cold 716,800-token TTFT.
- Temperature/sampling policy is part of a quality benchmark contract. Results from different policies are not interchangeable.
- GPU power values are sampled hardware-monitor values and may miss sub-sample transients.
- `p2pmark` was not run; no independent P2P bandwidth claim is made.
- The study does not establish results for Hopper, other Blackwell SKUs, other GPU counts, other EXL3 bitrates, BF16/NVFP4 model weights, or a newer runtime.

## 9. Evidence and integrity

All 31 July 23 benchmark outputs are retained under [`results/raw`](results/raw/) without editing. [`results/manifest.json`](results/manifest.json) records each timestamp, size, campaign phase, and SHA-256. [`results/SHA256SUMS`](results/SHA256SUMS) covers the reproducibility files. See [results/README.md](results/README.md) for verification.
