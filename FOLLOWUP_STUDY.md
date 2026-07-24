# GLM-5.2 EXL3 Follow-up Study

This report records the follow-up work performed on 23–24 July 2026 after the initial GLM-5.2 EXL3 v20 measurements. It separates measured results from interpretation, preserves rejected routes, and avoids presenting profile-specific gains as general improvements.

## Scope

The work used four NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs at a 600 W per-GPU power limit. It covered three related stages:

1. sparse-indexer memory and DCP profile experiments on the published v20 EXL3 runtime;
2. evaluation of the exact issue #34 RC2 runtime on its supported hybrid checkpoint;
3. a local merge of the EXL3 integration onto RC2, followed by MTP layer 78 conversion and quality validation.

Source ownership and exact contributor links are in [CREDITS.md](CREDITS.md). Raw measurements are indexed in [results/README.md](results/README.md).

## Stage 1: v20 EXL3 follow-up

### Adaptive sparse-indexer folding

The initial 786,432-token geometry encountered a transient sparse-indexer allocation barrier. The follow-up replaced a binary two-level fold switch with a bounded adaptive choice and selected a 256 MiB candidate-buffer cap. With that change, the tested profile exposed 999,424 configured tokens and completed a direct request with 998,800 prompt tokens plus 32 generated tokens.

This result applies to the v20 profile and direct request recorded in [`results/followup/`](results/followup/). It is not evidence that every prompt or sampling policy is reliable at that boundary. A 1,048,000-token experimental request executed, but its Estonia score was 8/10. A forced request above the model's native limit executed but failed structured retrieval. No usable-context claim is made above the native 1,048,576-token limit.

### Workload-specific DCP profiles

- DCP1 improved several decode cells but reduced the configured context ceiling to 262,144 tokens.
- An experimental DCP2 workspace profile improved prefill substantially and reached 373.6 aggregate tok/s at C8, but deterministic LAVD scored 8/10; it was rejected.
- MTP2 was competitive only in a narrow high-concurrency cell and was not selected over MTP3.

These are workload-specific observations, not universal rankings. Evidence remains under [`results/followup/`](results/followup/) with a neutral manifest at [`results/followup-manifest.json`](results/followup-manifest.json).

## Stage 2: issue #34 RC2 comparison

The exact issue #34 RC2 image was tested first on its supported MXFP8/NVFP4/NF3 hybrid checkpoint. It did not include the EXL3 integration and therefore served as an alternative runtime comparison rather than an immediate EXL3 upgrade.

The safe DCP2 profile completed a 179,017-token request, Estonia 10/10, and an automatic tool-call round trip. It delivered materially higher 64K/128K prefill than the prior v20 EXL3 profile, but it had a much lower context ceiling and weaker LAVD exactness in this sample. The faster DCP4/batch-5,120 candidate failed three of ten LAVD runs and was rejected.

Forced `tool_choice="required"` repeated identical tool calls until the output cap. That behavior became one of the explicit correction gates for the RC2+EXL3 merge.

Evidence: [`results/issue34/`](results/issue34/), [`results/issue34-manifest.json`](results/issue34-manifest.json), and [`configs/docker-compose.rc2-nf3.yml`](configs/docker-compose.rc2-nf3.yml).

## Stage 3: RC2+EXL3 merge

### Source integration

The local build rebased the EXL3 work from vLLM PR #139 and Sparkinfer PR #49 onto the source pins in issue #34 RC2. It also incorporated David Young's dual-plan Trellis prefill work from vLLM PR #163, already preserved in the EXL3 integration history.

Two narrow corrections were then applied:

- DCP projected/reduced workspace sizing and final partial-chunk handling in the merged MLA path;
- a structural-tag `required` grammar correction that stops after one valid required tool call instead of repeatedly generating identical calls.

The corresponding source diffs are published in [`patches/`](patches/). The diffs include upstream code and remain subject to the upstream projects' licenses.

### Controls

Three profiles were measured:

1. **RC2+EXL3 merged baseline:** merged runtime and original BF16 MTP layer 78.
2. **BF16 MTP78 quality build:** the corrected runtime with layer 78 retained in BF16.
3. **EXL3 MTP78 selected profile:** the same corrected runtime with layer 78 converted to EXL3.

The BF16 control separates runtime corrections from the effect of quantizing layer 78.

### Cold prefill

| Prompt tokens | Merged baseline | BF16 control | EXL3 MTP78 |
| ---: | ---: | ---: | ---: |
| 8K | 3,224 | 3,548 | **3,557** |
| 64K | 1,782 | **2,075** | 1,912 |
| 128K | 1,778 | **1,945** | 1,857 |

Values are tokens/second. The EXL3 MTP78 profile did not improve 64K or 128K prefill relative to the BF16 control.

### Sustained decode

| Context | Profile | C1 | C2 | C4 | C8 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Merged baseline | 101.7 | 163.7 | 237.5 | 372.0 |
| 0 | BF16 control | 100.8 | **166.9** | 244.8 | **372.7** |
| 0 | EXL3 MTP78 | 99.5 | 160.2 | **249.5** | 369.6 |
| 128K | Merged baseline | 102.4 | 159.7 | not fit | not fit |
| 128K | BF16 control | 100.8 | 160.2 | not fit | not fit |
| 128K | EXL3 MTP78 | **105.2** | **164.3** | **235.9** | not fit |

Values are aggregate output tokens/second. The primary measured benefit of the EXL3 MTP78 profile was the C4/128K capacity surface, not a uniform short-context throughput increase.

### Quality

| Profile | LAVD exact / near / fail | Estonia |
| --- | ---: | ---: |
| Merged baseline | 3 / 5 / 2 | not rerun |
| BF16 control | 4 / 5 / 1 | 10/10 |
| EXL3 MTP78 | 6 / 2 / 2 | 10/10 |

The ten-run LAVD sample is mixed: EXL3 MTP78 produced more exact responses but also one more hard failure than the BF16 control. Both the exact and failure counts are decision-relevant; neither is omitted.

### Tool behavior

Before the grammar correction, the required-tool test emitted 24 calls: 23 duplicates for Paris and a final empty object. After correction, the BF16 and EXL3 MTP78 profiles each emitted exactly one valid call and continued normally. The production smoke artifact records the selected profile's final response.

### Context boundary

The selected service was configured for 999,424 tokens. A direct request with 600,019 prompt tokens completed and returned `CONTEXT_OK`.

An artifact named `context900k` was generated by an attempted harness invocation, but the harness parser clamped it to 128K. It is excluded from this publication and is not treated as a 900K test. This study therefore claims only the observed 600,019-token RC2+EXL3 MTP78 smoke result.

### MTP layer 78 conversion

The conversion used 131,072 finite routed-activation rows and covered all 256 experts. Assembly replaced 768 BF16 expert weights with 12,288 rank-sliced EXL3 tensors. The total stored payload reduction was 15,662,567,424 bytes, approximately 3.65 GiB per TP rank.

The capture, validation, and assembly glue is under [`tools/mtp78/`](tools/mtp78/). The underlying encoder, LDLQ/Trellis method, MCG codebook, and ExLlamaV3 kernels are upstream work credited in [CREDITS.md](CREDITS.md). The published helpers do not include model weights or the calibration corpus.

## Selection rationale

The EXL3 MTP78 profile was selected for this host because it:

- retained the 999,424 configured service geometry;
- completed the observed 600,019-token direct smoke;
- passed Estonia 10/10;
- passed the corrected single-call required-tool gate;
- exposed a measured C4/128K decode cell unavailable to both BF16 profiles;
- reduced checkpoint payload by 15.66 GB total.

It was **not** selected because it was fastest or best in every metric. The BF16 control remained faster at 64K and 128K prefill, faster at zero-context C2/C8, and had fewer hard LAVD failures in this ten-run sample.

## Rejected or bounded claims

- No 900K RC2+EXL3 validation: the attempted harness run measured 128K.
- No universal quality gain: LAVD moved in both favorable and unfavorable directions.
- No universal speed gain: multiple BF16 cells remained faster.
- No ownership claim over the EXL3 encoder, Trellis/LDLQ/MCG methods, checkpoint, runtime integrations, or upstream fixes.
- No claim that a local image or locally assembled checkpoint is publicly downloadable.
- No generalization beyond four RTX PRO 6000 Blackwell GPUs and the recorded source/configuration pins.

## Primary evidence

- Merged baseline: [`results/raw/rc2-exl3-merged-baseline-prefill-20260724.json`](results/raw/rc2-exl3-merged-baseline-prefill-20260724.json), [`decode`](results/raw/rc2-exl3-merged-baseline-decode-20260724.json), [`LAVD`](results/raw/rc2-exl3-merged-baseline-lavd10-20260724.json).
- BF16 control: [`prefill`](results/raw/rc2-exl3-quality-bf16-prefill-20260724.json), [`decode`](results/raw/rc2-exl3-quality-bf16-decode-20260724.json), [`LAVD`](results/raw/rc2-exl3-quality-bf16-lavd10-20260724.json), [`Estonia`](results/raw/rc2-exl3-quality-bf16-estonia10-20260724.json), [`tool gate`](results/raw/rc2-exl3-quality-bf16-tool-gate-20260724.json).
- EXL3 MTP78: [`prefill`](results/raw/rc2-exl3-quality-mtp78-prefill-20260724.json), [`decode`](results/raw/rc2-exl3-quality-mtp78-decode-20260724.json), [`LAVD`](results/raw/rc2-exl3-quality-mtp78-lavd10-20260724.json), [`Estonia`](results/raw/rc2-exl3-quality-mtp78-estonia10-20260724.json), [`tool gate`](results/raw/rc2-exl3-quality-mtp78-tool-gate-20260724.json), [`600K smoke`](results/raw/rc2-exl3-quality-mtp78-context600k-smoke-20260724.json), and [`production smoke`](results/raw/rc2-exl3-production-smoke-20260724.json).
