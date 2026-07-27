# Result Archive

This directory preserves the evidence behind the GLM-5.2 EXL3 and MXFP8/NVFP4/NF3 hybrid Blackwell inference study.

Measurements were generated with [Local Inference Lab's `llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) against upstream model/runtime work credited in [`../CREDITS.md`](../CREDITS.md). Josh Cartu supplied and operated the four-GPU environment and published the archive. Raw harness field names and embedded methodology text belong to the cited harness.

## Integrity contract

- `raw/` contains original benchmark and direct-smoke outputs, including the RC2+EXL3 profile controls.
- `followup/` contains the 16 July 23 follow-up artifacts formerly published under promotional names; files were renamed for presentation, while harness payloads remain unchanged except the direct-observation record's neutral study label.
- `issue34/` contains 13 benchmark outputs plus one direct-observation record from the native RC2 comparison.
- `manifest.json`, `followup-manifest.json`, `issue34-manifest.json`, `rc2-exl3-manifest.json`, `optimization-manifest.json`, `hybrid-v20-manifest.json`, `campaign-20260726-manifest.json`, and `issue33-upgrade/manifest.json` record artifact provenance, sizes, and SHA-256 values.
- `SHA256SUMS` covers the publication's documentation, configurations, patches, helper scripts, manifests, and evidence.
- Failed candidates are retained when they establish a decision boundary.
- The invalid attempted `context900k` artifact is excluded because the harness actually clamped that invocation to 128K; it is not a 900K measurement.

Verify from the repository root:

```bash
sha256sum --check results/SHA256SUMS
```

## 26–27 July issue #33-era EXL3 archive

The final campaign adds 86 unedited temperature-one benchmark files under [`issue33-upgrade/`](issue33-upgrade/). It covers DCP1/DCP2/DCP4 baselines, topology-specific memory ceilings, exact request envelopes through 1,048,576 total tokens, communication and kernel controls, matched stock, final quality samples, and the restored production matrix.

Start with [`issue33-upgrade/README.md`](issue33-upgrade/README.md) or the plain-English report in [`../ISSUE33_STUDY.md`](../ISSUE33_STUDY.md).

| Question | Artifact |
| --- | --- |
| Final restored production matrix | [`production-final-dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-max512k-inference-temp1.json`](issue33-upgrade/production-final-dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-max512k-inference-temp1.json) |
| Matched old-stock matrix | [`stock-final-dcp2-g960-b4096-inference-temp1.json`](issue33-upgrade/stock-final-dcp2-g960-b4096-inference-temp1.json) |
| Exact DCP2 512K envelope | [`dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-native512k-full4096-temp1.json`](issue33-upgrade/dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-native512k-full4096-temp1.json) |
| Exact DCP4 1M envelope | [`dcp4-g977-owner0-chunk1-b3072-native1m-full4096-temp1.json`](issue33-upgrade/dcp4-g977-owner0-chunk1-b3072-native1m-full4096-temp1.json) |
| Final DCP2 quality | [`LAVD`](issue33-upgrade/final-dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-lavd20-c5-temp1.json) · [`Estonia`](issue33-upgrade/final-dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-estonia30-c5-temp1.json) |
| Final DCP4 quality | [`LAVD`](issue33-upgrade/final-dcp4-g978-owner0-chunk1-pf48-sh16-ckv140k-lavd20-c5-temp1.json) · [`Estonia`](issue33-upgrade/final-dcp4-g978-owner0-chunk1-pf48-sh16-ckv140k-estonia30-c5-temp1.json) |
| Machine-readable conclusion | [`study-summary.json`](issue33-upgrade/study-summary.json) |
| Per-file hashes and sizes | [`manifest.json`](issue33-upgrade/manifest.json) |

## 26 July campaign archive

### EXL3 v26 temperature-one profiles

| Profile | Inference matrix | LAVD | Estonia |
| --- | --- | --- | --- |
| DCP4 / batch 5,120 / max length 524,288 | [`8K–128K prefill and 0K–128K decode`](v26-tuning/winning-dcp4-b5120-temp1-inference.json) | [`C5: 5 exact / 5 near / 0 fail`](v26-tuning/winning-dcp4-b5120-temp1-lavd10.json) | [`C5: 10/10`](v26-tuning/winning-dcp4-b5120-temp1-estonia10.json) |
| DCP2 / batch 4,096 / max length 300,000 | [`8K–128K prefill and 0K–128K decode`](v26-tuning/winning-dcp2-b4096-temp1-inference.json) | [`C5: 6 / 3 / 1`](v26-tuning/winning-dcp2-b4096-temp1-lavd10.json) | [`C3: 10/10`](v26-tuning/winning-dcp2-b4096-temp1-estonia10.json) |
| DCP1 / batch 4,096 / max length 180,000 | [`8K–128K prefill and 0K–128K decode`](v26-tuning/winning-dcp1-b4096-temp1-inference.json) | [`C1: 3 / 6 / 1`](v26-tuning/winning-dcp1-b4096-temp1-lavd10-c1-control.json) · [`C5 boundary: 1 / 0 / 9`](v26-tuning/winning-dcp1-b4096-temp1-lavd10.json) | [`C1: 10/10`](v26-tuning/winning-dcp1-b4096-temp1-estonia10.json) |

Every row used explicit temperature `1.0`; inference used exact prompt-token targeting and 15-second sustained-decode windows. LAVD used ten measured runs with a 40,000-token ceiling. Estonia used capacity-safe concurrency: C5 for DCP4, C3 for DCP2, and C1 for DCP1. DCP1's C5 failure is retained as a measured concurrency boundary rather than hidden.

The remainder of [`v26-tuning/`](v26-tuning/) preserves the tuning sequence, rejected candidates, tool/context gates, sampled quality controls, and final service evidence that led to these three Pareto points.

### GPU memory-utilization ceiling sweep

| Profile | Highest passing boundary | First confirmed OOM | Saturation evidence |
| --- | --- | --- | --- |
| DCP4 | [`0.96875`, 512K input + 4,096 output](v26-tuning/memory-ceiling-dcp4-g96875-stress.json) | [`0.96900`](v26-tuning/memory-ceiling-dcp4-g9690-stress.json) | [`C2`, 796,672 / 801,792 scheduled KV tokens](v26-tuning/memory-ceiling-dcp4-g96875-c2-saturation.json) |
| DCP2 | [`0.96750`, 294,912 input + 4,096 output](v26-tuning/memory-ceiling-dcp2-g9675-stress.json) | [`0.96775`](v26-tuning/memory-ceiling-dcp2-g96775-stress.json) | [`C2`, 479,232 / 482,944 scheduled KV tokens](v26-tuning/memory-ceiling-dcp2-g9675-c2-saturation.json) |
| DCP1 | [`0.98250`, 174,080 input + 4,096 output](v26-tuning/memory-ceiling-dcp1-g9825-stress.json) | [`0.98275`](v26-tuning/memory-ceiling-dcp1-g98275-stress.json) | [`C2`, 315,392 / 322,496 scheduled KV tokens](v26-tuning/memory-ceiling-dcp1-g9825-c2-saturation.json) |

The machine-readable [`ceiling summary`](v26-tuning/memory-ceiling-summary-20260726.json) records the protocol, closest pass/fail brackets, observed OOM allocations, all 28 harness artifacts with SHA-256 provenance, and the final restoration of DCP4/0.96 production. Engine startup alone materially overstated the usable ceiling.

### Provenance-hardened MTP78 and vision evidence

| Question | Artifact |
| --- | --- |
| What was accepted for the upstream MTP78 text canary, and what boundaries remain? | [`integration decision`](mtp78-upstream-integration-decision.json) |
| How were source, overlay, runtime, and service inputs pinned? | [`runtime provenance`](mtp78-upstream-runtime-provenance.json) |
| Did BF16 and Trellis match at the target-only first step? | [`target-only comparison`](mtp78-upstream-target-only-comparison.json) · [`T0 fingerprint`](mtp78-upstream-t0-first-token-fingerprint.json) · [`T1 fingerprint`](mtp78-upstream-t1-first-token-fingerprint.json) |
| What were the matched BF16/Trellis acceptance and throughput results? | [`T2/T3 comparison`](mtp78-upstream-t2-t3-comparison.json) · [`Trellis full bench`](mtp78-upstream-t3-trellis-full-bench.json) |
| Did the retained text profile pass long context and quality controls? | [`600,019-token smoke`](mtp78-upstream-t3-fixed3904-600019-smoke.json) · [`LAVD`](mtp78-upstream-t3-lavd3.json) · [`Estonia`](mtp78-upstream-t3-estonia3-temp0.json) |
| Did MTP78 change GLM-5.2-Vision semantics or acceptance? | [`matched canary comparison`](vision-canary-comparison.json) |
| Why did the 200K vision gate fail, and was recalibration triggered? | [`capacity decision`](vision-acceptance-capacity-decision.json) |

The raw archive also contains every generated synthetic canary image, request, response body, tokenizer response, Prometheus snapshot, matched text control, and capacity probe. Responses were persisted before decoding or grading. The shared two-image B/A ordering failure exists in all MTP modes; the long-context failure reproduces with MTP disabled and text-only input, so neither is attributed to the Trellis MTP78 overlay.

## RC2+EXL3 primary artifacts

| Question | Artifact |
| --- | --- |
| What is the merged baseline prefill/decode result? | [`prefill`](raw/rc2-exl3-merged-baseline-prefill-20260724.json) · [`decode`](raw/rc2-exl3-merged-baseline-decode-20260724.json) |
| What was the merged baseline LAVD distribution? | [`3 exact / 5 near / 2 fail`](raw/rc2-exl3-merged-baseline-lavd10-20260724.json) |
| What did the corrected BF16 MTP78 control measure? | [`prefill`](raw/rc2-exl3-quality-bf16-prefill-20260724.json) · [`decode`](raw/rc2-exl3-quality-bf16-decode-20260724.json) |
| Did the BF16 control pass quality and required-tool gates? | [`LAVD`](raw/rc2-exl3-quality-bf16-lavd10-20260724.json) · [`Estonia`](raw/rc2-exl3-quality-bf16-estonia10-20260724.json) · [`tool gate`](raw/rc2-exl3-quality-bf16-tool-gate-20260724.json) |
| What did the EXL3 MTP78 profile measure? | [`prefill`](raw/rc2-exl3-quality-mtp78-prefill-20260724.json) · [`decode`](raw/rc2-exl3-quality-mtp78-decode-20260724.json) |
| Did EXL3 MTP78 pass quality and required-tool gates? | [`LAVD`](raw/rc2-exl3-quality-mtp78-lavd10-20260724.json) · [`Estonia`](raw/rc2-exl3-quality-mtp78-estonia10-20260724.json) · [`tool gate`](raw/rc2-exl3-quality-mtp78-tool-gate-20260724.json) |
| What long-context request was actually observed? | [`600,019-token CONTEXT_OK smoke`](raw/rc2-exl3-quality-mtp78-context600k-smoke-20260724.json) |
| Did the selected service complete its final API smoke? | [`production smoke`](raw/rc2-exl3-production-smoke-20260724.json) |

The interpretation and mixed-result boundaries are in [`../FOLLOWUP_STUDY.md`](../FOLLOWUP_STUDY.md).

## Accuracy-preserving optimization

| Question | Artifact |
| --- | --- |
| What were the binding gates and final decision? | [`gates`](optimization/optimization-gates-20260724.json) · [`summary`](optimization/optimization-summary-20260724.json) |
| What was the fresh 3,072-token baseline? | [`prefill`](optimization/opt-baseline-prefill-20260724.json) · [`decode`](optimization/opt-baseline-decode-20260724.json) |
| What does deployed batch 5,120 measure? | [`prefill`](optimization/opt-production-b5120-prefill-20260724.json) · [`decode`](optimization/opt-production-b5120-decode-20260724.json) |
| Did deployed geometry pass quality and behavior gates? | [`LAVD`](optimization/opt-dcp4-b5120-lavd10-20260724.json) · [`Estonia`](optimization/opt-dcp4-b5120-estonia10-20260724.json) · [`tool`](optimization/opt-dcp4-b5120-tool-gate-20260724.json) · [`600K`](optimization/opt-dcp4-b5120-context600k-smoke-20260724.json) |
| Why was faster DCP2 rejected? | [`prefill`](optimization/opt-dcp2-b4096-k4800-prefill-20260724.json) · [`decode`](optimization/opt-dcp2-b4096-k4800-decode-20260724.json) · [`LAVD 4/5/1`](optimization/opt-dcp2-b4096-k4800-lavd10-20260724.json) |
| What MTP and scheduler boundaries were measured? | [`MTP2`](optimization/opt-b5120-mtp2-decode-20260724.json) · [`MTP4`](optimization/opt-b5120-mtp4-decode-20260724.json) · [`batch 5,632`](optimization/opt-b5632-prefill-20260724.json) |

The optimization archive contains successful candidates and the measured decision boundary. Batch 6,144 and 8,192 failed before a harness JSON could be written; their observed CUDA OOM allocation details are preserved in the optimization summary.

## July 23 follow-up archive

| Question | Artifact |
| --- | --- |
| What is the 999,424-token v20 profile decode matrix? | [`exl3-followup-production-b3072-decode-20260723.json`](followup/exl3-followup-production-b3072-decode-20260723.json) |
| What is its exact-token prefill result? | [`exl3-followup-production-b3072-prefill-20260723.json`](followup/exl3-followup-production-b3072-prefill-20260723.json) |
| Did that profile pass LAVD and Estonia? | [`LAVD`](followup/exl3-followup-b3072-quality-lavd10-20260723.json) · [`Estonia`](followup/exl3-followup-b3072-quality-estonia10-20260723.json) |
| What native/beyond-native requests ran? | [`direct evidence`](followup/exl3-followup-direct-evidence-20260723.json) |
| How did DCP1 decode? | [`DCP1`](followup/exl3-followup-dcp1-decode-20260723.json) |
| Why was the faster DCP2 workspace route rejected? | [`prefill`](followup/exl3-followup-dcp2-workspace-prefill-20260723.json) · [`decode`](followup/exl3-followup-dcp2-workspace-decode-matrix-20260723.json) · [`LAVD 8/10`](followup/exl3-followup-dcp2-workspace-lavd10-20260723.json) |

## NF3 hybrid v20 optimization

| Question | Artifact |
| --- | --- |
| What was predeclared, and what execution deviations were found? | [`gates`](hybrid-v20/hybrid-optimization-gates-20260725.json) · [`protocol amendment`](hybrid-v20/hybrid-protocol-amendment-20260725.json) |
| What is the corrected machine-readable decision? | [`scorecard`](hybrid-v20/hybrid-v20-scorecard-20260725.json) |
| What did the exact attached balanced control measure? | [`8K–128K prefill`](hybrid-v20/hybrid-attached-baseline-prefill-exact-20260725.json) · [`240K/400K prefill`](hybrid-v20/hybrid-sota-final-prefill-long-exact-20260725.json) · [`decode`](hybrid-v20/hybrid-attached-baseline-decode-20260725.json) |
| What did the explicit-temperature-zero reasoning gates measure? | [`LAVD 1/5/4`](hybrid-v20/hybrid-sota-final-lavd10-explicit-t0-40k-20260725.json) · [`GSM8K 98/100`](hybrid-v20/hybrid-attached-baseline-gsm8k100-20260725.json) · [`Estonia 3/10`](hybrid-v20/hybrid-sota-final-estonia10-explicit-t0-40k-20260725.json) |
| Did tool, context, stress, and post-stress health checks pass? | [`tool`](hybrid-v20/hybrid-sota-stable-tool-gate-20260725.json) · [`450K context, 512-token stress, and health`](hybrid-v20/hybrid-sota-final-context450k-health-20260725.json) |
| What are the matched DCP2/180K performance results? | [`prefill`](hybrid-v20/hybrid-attached-dcp2-180k-prefill-exact-20260725.json) · [`30-second temperature-zero decode`](hybrid-v20/hybrid-sota-dcp2-180k-decode30-t0-20260725.json) |
| What are the matched DCP1/90K performance results? | [`prefill`](hybrid-v20/hybrid-attached-dcp1-90k-prefill-exact-20260725.json) · [`30-second temperature-zero decode`](hybrid-v20/hybrid-sota-dcp1-90k-decode30-t0-20260725.json) |
| Why was reordered full-context production rejected? | [`candidate decode`](hybrid-v20/hybrid-attached-order3120-decode20-20260725.json) · [`production repeat`](hybrid-v20/hybrid-sota-production-decode30-20260725.json) · [`60-second confirmation`](hybrid-v20/hybrid-sota-production-decode32k-c4-confirm-20260725.json) |
| What did the single newer-RC2 screening run show? | [`prefill`](hybrid-v20/hybrid-rc2-b3072-g978-prefill-exact-20260725.json) · [`decode`](hybrid-v20/hybrid-rc2-b3072-g978-decode15-20260725.json) |
| How is the checkpoint pinned? | [`revision and small-file hashes`](hybrid-v20/checkpoint-provenance-20260725.json) |
| Which startup/OOM boundaries lack raw harness output? | [`operator observations`](hybrid-v20/startup-failure-observations-20260725.json) · [`absorb-BMM prefill`](hybrid-v20/hybrid-old-b3072-spec1-absorb1-prefill-exact-20260725.json) · [`absorb-BMM decode`](hybrid-v20/hybrid-old-b3072-spec1-absorb1-decode20-20260725.json) |

The unchanged DCP4 control preserves the requested 479,744-token operational geometry but failed the corrective temperature-zero LAVD and Estonia gates. Earlier omitted-sampling completion-profile artifacts and unmatched decode windows remain in the raw archive and are explicitly marked superseded by the protocol amendment. DCP1 and DCP2 are performance/capacity Pareto points and were not separately quality-gated. The full interpretation is in [`../HYBRID_STUDY.md`](../HYBRID_STUDY.md).

## Native issue #34 RC2 comparison

| Question | Artifact |
| --- | --- |
| What image, revisions, settings, context, and tool observations were recorded? | [`rc2-direct-evidence-20260724.json`](issue34/rc2-direct-evidence-20260724.json) |
| What is the safe DCP2 prefill/decode result? | [`prefill`](issue34/rc2-nf3-dcp2-mtp3-b3072-prefill-20260724.json) · [`decode`](issue34/rc2-nf3-dcp2-mtp3-b3072-decode-20260724.json) |
| Did DCP2 pass its gates? | [`LAVD`](issue34/rc2-nf3-dcp2-mtp3-b3072-lavd10-20260724.json) · [`Estonia`](issue34/rc2-nf3-dcp2-mtp3-b3072-l180k-estonia10-20260724.json) |
| Why was DCP4/batch-5,120 rejected? | [`3 exact / 4 near / 3 fail`](issue34/rc2-nf3-dcp4-mtp3-b5120-g970-lavd10-20260724.json) |

## Original v2/v20 archive

Historical raw filenames containing `winning` are retained because they are byte-for-byte benchmark output names, not current editorial claims. The original profile matrix, capacity checks, sampling-policy interaction, and rejection sequence remain documented in [`../METHODOLOGY.md`](../METHODOLOGY.md) and indexed by [`manifest.json`](manifest.json).

## JSON structure

Decode/prefill outputs generally include `metadata`, `startup_diagnostics`, `hardware_run_summary`, `event_log`, per-cell measurements, summary tables, and harness-authored methodology. Completion-profile outputs generally include run policy, selected/all summaries, per-run answers and scores, timing/token distributions, and hardware summaries.

## Scope and privacy

Raw benchmark files include the study hostname (`rasputin`), Linux/driver versions, PCI bus IDs, topology, localhost endpoint, and sampled hardware state because those facts affect reproducibility. They contain no model weights, API credentials, access tokens, private keys, or container layers.
