# Result Archive

This directory preserves the evidence behind the GLM-5.2 EXL3 Blackwell study.

> Attribution: measurements were generated with [Local Inference Lab's `llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) against upstream model/runtime work credited in [`../CREDITS.md`](../CREDITS.md). Josh Cartu supplied and operated the four-GPU campaign environment and published the resulting study archive. Raw benchmark field names and embedded methodology text originate from the cited harness.

## Integrity contract

- `raw/` contains **31 byte-for-byte JSON copies** produced on 23 July 2026.
- `manifest.json` records each raw file's campaign phase, timestamp, byte count, and SHA-256.
- `SHA256SUMS` covers the raw artifacts, manifest, public-safe configurations, and tool-call proof.
- `tool-call-smoke.json` is a fresh, narrowed end-to-end observation of the final service's OpenAI-compatible tool-call contract.
- Raw evidence was not rewritten to improve presentation or remove failed candidates.

Verify from the repository root:

```bash
sha256sum --check results/SHA256SUMS
```

## Primary final artifacts

| Question | Artifact |
| --- | --- |
| What is the final zero/32K/128K decode matrix? | [`exl3-v20-winning-decode-matrix-20260723.json`](raw/exl3-v20-winning-decode-matrix-20260723.json) |
| What is the exact-token 8K/64K/128K prefill result? | [`exl3-v20-winning-prefill-exact-20260723.json`](raw/exl3-v20-winning-prefill-exact-20260723.json) |
| Did exact 716,800-token input fit and decode? | [`exl3-v20-mtp3-specdecode-kvrope-700k-exact-decode-20260723.json`](raw/exl3-v20-mtp3-specdecode-kvrope-700k-exact-decode-20260723.json) |
| Did the final sampled Estonia profile pass? | [`exl3-v20-final-estonia10-sampled-20260723.json`](raw/exl3-v20-final-estonia10-sampled-20260723.json) |
| Did deterministic LAVD pass with enough output budget? | [`exl3-v20-winning-lavd10-24k-20260723.json`](raw/exl3-v20-winning-lavd10-24k-20260723.json) |
| Did OpenAI-compatible tool calling work? | [`tool-call-smoke.json`](tool-call-smoke.json) |

## Baseline and planned-prefill stage

| Artifact | Purpose |
| --- | --- |
| `exl3-v2-baseline-decode-20260723.json` | Original v2 sustained-decode baseline, C1–C8 |
| `exl3-v2-baseline-prefill-20260723.json` | Original exact-token cold-prefill baseline |
| `exl3-v2-trellis-m2-decode-20260723.json` | v2 planned-Trellis decode check |
| `exl3-v2-trellis-m2-prefill-20260723.json` | v2 planned-Trellis prefill candidate |
| `exl3-v2-trellis-m2-block48-prefill-20260723.json` | Prefill block-48 candidate |
| `exl3-v2-trellis-m2-block32-prefill-20260723.json` | Prefill block-32 candidate |

The dual-plan prefill implementation belongs to David Young's [vLLM PR #163](https://github.com/local-inference-lab/vllm/pull/163), later incorporated into Brandon Music's [vLLM PR #139](https://github.com/local-inference-lab/vllm/pull/139) with authorship preserved.

## v20 MTP and speculative-decode sweep

| Artifact | Purpose |
| --- | --- |
| `exl3-v20-mtp3-prefill-20260723.json` | Early v20 prefill check |
| `exl3-v20-mtp3-decode-20260723.json` | Early v20 MTP3 decode check |
| `exl3-v20-mtp2-specdecode-decode-20260723.json` | MTP2 with retained speculative extension path |
| `exl3-v20-mtp3-specdecode-decode-20260723.json` | MTP3 candidate |
| `exl3-v20-mtp4-specdecode-decode-20260723.json` | MTP4 candidate |
| `exl3-v20-mtp5-specdecode-decode-20260723.json` | MTP5 candidate |

## KV/capacity, scheduler, and draft-backend sweep

| Artifact group | Purpose |
| --- | --- |
| `exl3-v20-mtp3-specdecode-kvrope-decode-20260723.json` | MTP3 + speculative extension + KV-FP8-RoPE candidate |
| `...-700k-decode.json`, `...-700k-decode-retry.json` | Early capacity calibration attempts |
| `...-700k-exact-decode.json` | Primary exact 716,800-token capacity proof |
| `...-900k-decode.json` | Retained failed boundary attempt; no 1M claim |
| `...-batch2k-decode.json` | 2,048 batched-token candidate |
| `...-batch4k-decode.json`, `...-batch4k-decode-retry.json` | Two rejected 4,096-token runs |
| `...-drafttriton-control-decode.json` | Triton draft-MoE control |
| `...-draftb12x-decode.json` | B12X draft-MoE candidate |

## Final quality artifacts

| Artifact | Policy and result |
| --- | --- |
| `exl3-v20-winning-lavd10-16k-20260723.json` | Deterministic LAVD, 16K cap: 9/10 acceptable, one cap hit |
| `exl3-v20-winning-lavd10-24k-20260723.json` | Deterministic LAVD, 24K cap: 10/10 acceptable, zero cap hits |
| `exl3-v20-winning-estonia10-20260723.json` | Greedy Estonia, MTP3/KV on: 4/10 |
| `exl3-v20-kvrope0-estonia10-20260723.json` | Greedy Estonia, KV-FP8-RoPE off: 4/10 |
| `exl3-v20-mtp1-estonia10-20260723.json` | Greedy Estonia, MTP1/KV off: 0/10 |
| `exl3-v20-final-estonia10-sampled-20260723.json` | Default-sampled final Estonia: 10/10, zero cap hits |

These files are deliberately presented together. Omitting the deterministic failures would conceal the sampling-policy interaction; presenting only those failures would conceal the successful final policy. See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the interpretation boundary.

## JSON structure

Decode/prefill files generally contain:

- `metadata`: harness/runtime/mode and measurement arguments;
- `startup_diagnostics`: host, NVIDIA, topology, endpoint, and argument discovery;
- `hardware_run_summary`: sampled utilization, temperature, power, VRAM, and PCIe counters;
- `event_log`: warmups, cell starts/completions, capacity decisions, and errors;
- `prefill`: per-context cold-prefill results;
- `results` and `summary_table`: per-cell detailed and matrix decode output;
- `methodology`: harness-authored definitions for each metric.

Completion-stat/profile files generally contain:

- `metadata`: profile, prompt/dataset identity, policy, run count, and scorer;
- `selected_summary` / `all_summary`: completion, correctness, token, TTFT, and throughput distributions;
- `runs`: per-run final answer, score, token counts, latency, and finish reason;
- `hardware_run_summary` and `methodology`.

## Scope and privacy

The raw benchmark files include the campaign host name (`rasputin`), Linux/driver versions, PCI bus IDs, topology, and sampled hardware state because those facts matter to reproducibility. They contain no model weights, API credentials, access tokens, or container layers.

The endpoint value is localhost and does not identify a remotely accessible service.
