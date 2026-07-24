# Result Archive

This directory preserves the evidence behind the GLM-5.2 EXL3 Blackwell inference study.

Measurements were generated with [Local Inference Lab's `llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) against upstream model/runtime work credited in [`../CREDITS.md`](../CREDITS.md). Josh Cartu supplied and operated the four-GPU environment and published the archive. Raw harness field names and embedded methodology text belong to the cited harness.

## Integrity contract

- `raw/` contains original benchmark and direct-smoke outputs, including the RC2+EXL3 profile controls.
- `followup/` contains the 16 July 23 follow-up artifacts formerly published under promotional names; files were renamed for presentation, while harness payloads remain unchanged except the direct-observation record's neutral study label.
- `issue34/` contains 13 benchmark outputs plus one direct-observation record from the native RC2 comparison.
- `manifest.json`, `followup-manifest.json`, `issue34-manifest.json`, and `rc2-exl3-manifest.json` record artifact provenance, sizes, and SHA-256 values.
- `SHA256SUMS` covers the publication's documentation, configurations, patches, helper scripts, manifests, and evidence.
- Failed candidates are retained when they establish a decision boundary.
- The invalid attempted `context900k` artifact is excluded because the harness actually clamped that invocation to 128K; it is not a 900K measurement.

Verify from the repository root:

```bash
sha256sum --check results/SHA256SUMS
```

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

## July 23 follow-up archive

| Question | Artifact |
| --- | --- |
| What is the 999,424-token v20 profile decode matrix? | [`exl3-followup-production-b3072-decode-20260723.json`](followup/exl3-followup-production-b3072-decode-20260723.json) |
| What is its exact-token prefill result? | [`exl3-followup-production-b3072-prefill-20260723.json`](followup/exl3-followup-production-b3072-prefill-20260723.json) |
| Did that profile pass LAVD and Estonia? | [`LAVD`](followup/exl3-followup-b3072-quality-lavd10-20260723.json) · [`Estonia`](followup/exl3-followup-b3072-quality-estonia10-20260723.json) |
| What native/beyond-native requests ran? | [`direct evidence`](followup/exl3-followup-direct-evidence-20260723.json) |
| How did DCP1 decode? | [`DCP1`](followup/exl3-followup-dcp1-decode-20260723.json) |
| Why was the faster DCP2 workspace route rejected? | [`prefill`](followup/exl3-followup-dcp2-workspace-prefill-20260723.json) · [`decode`](followup/exl3-followup-dcp2-workspace-decode-matrix-20260723.json) · [`LAVD 8/10`](followup/exl3-followup-dcp2-workspace-lavd10-20260723.json) |

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
