# Result Archive

This directory preserves the evidence behind the GLM-5.2 EXL3 Blackwell study.

> Attribution: measurements were generated with [Local Inference Lab's `llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) against upstream model/runtime work credited in [`../CREDITS.md`](../CREDITS.md). Josh Cartu supplied and operated the four-GPU campaign environment and published the resulting study archive. Raw benchmark field names and embedded methodology text originate from the cited harness.

## Integrity contract

- `raw/` contains **31 byte-for-byte JSON copies** from the original 23 July campaign.
- `breakthrough/` contains **16 byte-for-byte JSON copies** from the EXL3 follow-on deep dive.
- `issue34/` contains **13 byte-for-byte benchmark JSON copies** plus one direct-observation record from the 24 July RC2 campaign.
- `manifest.json`, `breakthrough-manifest.json`, and `issue34-manifest.json` record provenance for the three artifact sets.
- `SHA256SUMS` covers all artifact sets, manifests, public-safe runtime/configuration files, documentation, and tool-call proof.
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

## Breakthrough extension

| Question | Artifact |
| --- | --- |
| What is the final 999,424-token production decode matrix? | [`exl3-breakthrough-production-b3072-decode-20260723.json`](breakthrough/exl3-breakthrough-production-b3072-decode-20260723.json) |
| What is the final exact-token prefill result? | [`exl3-breakthrough-production-b3072-prefill-20260723.json`](breakthrough/exl3-breakthrough-production-b3072-prefill-20260723.json) |
| Did production LAVD and Estonia pass? | [`LAVD`](breakthrough/exl3-breakthrough-b3072-quality-lavd10-20260723.json) · [`Estonia`](breakthrough/exl3-breakthrough-b3072-quality-estonia10-20260723.json) |
| What direct native/beyond-native capacity and retrieval requests ran? | [`exl3-breakthrough-direct-evidence-20260723.json`](breakthrough/exl3-breakthrough-direct-evidence-20260723.json) |
| How fast was DCP1? | [`exl3-breakthrough-dcp1-decode-20260723.json`](breakthrough/exl3-breakthrough-dcp1-decode-20260723.json) |
| How fast was the experimental DCP2 workspace profile? | [`prefill`](breakthrough/exl3-breakthrough-dcp2-workspace-prefill-20260723.json) · [`decode`](breakthrough/exl3-breakthrough-dcp2-workspace-decode-matrix-20260723.json) |
| Why was DCP2 rejected? | [`LAVD 8/10`](breakthrough/exl3-breakthrough-dcp2-workspace-lavd10-20260723.json) · [`Estonia 10/10`](breakthrough/exl3-breakthrough-dcp2-workspace-estonia10-20260723.json) |
| What happened at the DCP4 native edge? | [`LAVD 10/10`](breakthrough/exl3-breakthrough-final-no-workspace-lavd10-20260723.json) · [`Estonia 8/10`](breakthrough/exl3-breakthrough-final-safe-estonia10-20260723.json) |

The interpretation, claim boundaries, and source changes are documented in [`../BREAKTHROUGH_CAMPAIGN.md`](../BREAKTHROUGH_CAMPAIGN.md).

## Issue #34 RC2 follow-on

| Question | Artifact |
| --- | --- |
| What immutable image, revisions, service settings, direct context, and tool observations were recorded? | [`rc2-direct-evidence-20260724.json`](issue34/rc2-direct-evidence-20260724.json) |
| What is the safe DCP2 exact-token prefill result? | [`rc2-nf3-dcp2-mtp3-b3072-prefill-20260724.json`](issue34/rc2-nf3-dcp2-mtp3-b3072-prefill-20260724.json) |
| What is the safe DCP2 decode matrix? | [`rc2-nf3-dcp2-mtp3-b3072-decode-20260724.json`](issue34/rc2-nf3-dcp2-mtp3-b3072-decode-20260724.json) |
| Did DCP2 pass quality gates? | [`LAVD 10/10 acceptable`](issue34/rc2-nf3-dcp2-mtp3-b3072-lavd10-20260724.json) · [`Estonia 10/10`](issue34/rc2-nf3-dcp2-mtp3-b3072-l180k-estonia10-20260724.json) |
| Why is the first Estonia file retained? | [`147K configured-limit rejection`](issue34/rc2-nf3-dcp2-mtp3-b3072-estonia10-20260724.json); it is an HTTP 400 context guard, not a model-quality failure |
| How did DCP1 decode? | [`rc2-nf3-dcp1-mtp3-b3072-decode-20260724.json`](issue34/rc2-nf3-dcp1-mtp3-b3072-decode-20260724.json) |
| What happened across DCP4 scheduler budgets? | [`batch 3,072 prefill`](issue34/rc2-nf3-dcp4-mtp3-b3072-prefill-20260724.json) · [`decode`](issue34/rc2-nf3-dcp4-mtp3-b3072-decode-20260724.json) · [`batch 4,096 prefill`](issue34/rc2-nf3-dcp4-mtp3-b4096-g965-prefill-20260724.json) · [`decode`](issue34/rc2-nf3-dcp4-mtp3-b4096-g965-decode-20260724.json) · [`batch 5,120 prefill`](issue34/rc2-nf3-dcp4-mtp3-b5120-g970-prefill-20260724.json) · [`decode`](issue34/rc2-nf3-dcp4-mtp3-b5120-g970-decode-20260724.json) |
| Why was the fastest DCP4 prefill candidate rejected? | [`LAVD 3/4/3 exact/near/fail`](issue34/rc2-nf3-dcp4-mtp3-b5120-g970-lavd10-20260724.json) |

The interpretation and production decision are documented in the [issue #34 campaign section](../BREAKTHROUGH_CAMPAIGN.md#issue-34-rc2-follow-on-24-july-2026). Checksums and byte counts are in [`issue34-manifest.json`](issue34-manifest.json).

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
