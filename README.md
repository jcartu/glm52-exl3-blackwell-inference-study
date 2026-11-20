# GLM-5.2 Blackwell Inference Study: EXL3 and NF3 Hybrid

A reproducible record of GLM-5.2 inference measurements collected on 23–27 July 2026 with four NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs. The study covers a rank-sliced 3.0 bpw EXL3/Trellis checkpoint from the original Gilded Gnosis v20 evaluation through an issue #34 RC2 rebase, experimental EXL3-quantized MTP layer 78, a topology-aware v26 Pareto study, matched GLM-5.2-Vision canaries, a bounded MXFP8/NVFP4/NF3 hybrid study, and the final issue #33-era EXL3 runtime and flag audit.

This is an engineering study, not a claim of a new model, quantization method, or general performance record. Results apply only to the pinned software, checkpoint variants, hardware topology, and benchmark settings recorded here.

> [!IMPORTANT]
> This repository contains study documentation, public-safe configurations, source patches, helper scripts, and measured artifacts. It does **not** contain or claim ownership of GLM-5.2 weights, the EXL3 or NF3-hybrid checkpoints, vLLM, Sparkinfer, ExLlamaV3/Trellis, Gilded Gnosis, or referenced container images. See [CREDITS.md](CREDITS.md) for the complete ownership and contribution ledger.

## Latest result: issue #33-era EXL3 production study

The 26–27 July follow-up rebuilt the 3.0 bpw EXL3 service on the exact issue #33 candidate source cut, tested DCP1/DCP2/DCP4, established real memory-failure boundaries, audited every exposed EXL3/DCP performance control, and reran matched temperature-one inference, LAVD, and Estonia gates.

For readers who do not work with inference runtimes every day, the complete plain-English report is:

**[GLM-5.2 EXL3 on Four RTX PRO 6000 GPUs: Issue #33-Era Study](ISSUE33_STUDY.md)**

The selected normal-production profile is now **TP4/DCP2/MTP3**, not the earlier DCP4 service:

| Metric | Matched old stock | Selected issue33-era profile | Change |
| --- | ---: | ---: | ---: |
| Verified/configured request envelope | 300,000 configured | **512,000 completed total** | +212,000 configured/verified tokens |
| Reported KV capacity | 440,448 | **513,536** | **+16.6%** |
| 8K cold prefill | **3,746 tok/s** | 3,644 tok/s | -2.7% |
| 64K cold prefill | 3,155 tok/s | **3,380 tok/s** | **+7.1%** |
| 128K cold prefill | 3,023 tok/s | **3,140 tok/s** | **+3.9%** |
| Common-cell decode geometric mean | baseline | effectively matched | **-0.15%** |

The selected service completed an exact 507,904-token prompt plus 4,096 generated tokens. DCP4 remains the explicit maximum-context option and completed a separate 1,048,576-token total request envelope.

Quality at explicit temperature `1.0` showed no statistically detectable difference in the exact matched DCP2 comparison:

- selected DCP2 LAVD: 19/20 acceptable versus matched stock 20/20, Fisher `p = 1.0`;
- selected DCP2 Estonia: 29/30 versus matched stock 30/30, Fisher `p = 1.0`;
- DCP4 results are reported descriptively because the archive has no matched old-stock DCP4 quality control.

The largest performance discovery was raising transient full-CKV gather capacity from 16,384 to 140,000 tokens. It improved 64K/128K prefill by 15.8%/12.6% under DCP2 and 49.1%/50.0% under DCP4.

> [!NOTE]
> This campaign tested the issue #33 `0c79e41`/`e603f74` source cut with the pinned EXL3 PR #139/#49 heads. Issue #33 later published a different `sic3828fd`/`r4` image. The numbers in this repository do not describe that later image.

The public-safe measured serving config is [`configs/docker-compose.exl3-v20.yml`](configs/docker-compose.exl3-v20.yml), the derivative build recipe is [`configs/Dockerfile.issue33-exl3`](configs/Dockerfile.issue33-exl3), and all 86 unedited temperature-one benchmark artifacts are indexed under [`results/issue33-upgrade/`](results/issue33-upgrade/).

## EXL3 v26 temperature-one Pareto matrix

The 26 July v26 matrix used `llm-inference-bench` v0.4.28, temperature `1.0`, exact prompt-token targeting, 15-second sustained-decode cells, MTP3, and the same four-GPU host. Capacity-limited cells were skipped rather than reported as throughput.

| Profile | Intended role | Max model length | KV budget | 8K prefill | 64K prefill | 128K prefill | 0K C1 decode | 0K C8 decode |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DCP4 / batch 5,120 | balanced production | 524,288 | 705,024 | 2,995 | 1,862 | 1,777 | 97.5 | 336.0 |
| DCP2 / batch 4,096 | concurrent throughput | 300,000 | 441,600 | 3,756 | 3,144 | 3,025 | 102.9 | **407.3** |
| DCP1 / batch 4,096 | single-stream throughput | 180,000 | 260,480 | **3,939** | **4,039** | **3,973** | **116.0** | 372.4 |

All rates are tokens/second. The full decode matrix also covers 32K and 128K input contexts in [`results/v26-tuning/`](results/v26-tuning/).

| Profile | LAVD concurrency | LAVD exact / near / fail | Max-token hits | Estonia concurrency | Estonia |
| --- | ---: | ---: | ---: | ---: | ---: |
| DCP4 | 5 | **5 / 5 / 0** | 0 | 5 | **10/10** |
| DCP2 | 5 | 6 / 3 / 1 | 0 | 3 | **10/10** |
| DCP1 single-stream control | 1 | 3 / 6 / 1 | 0 | 1 | **10/10** |
| DCP1 concurrency boundary | 5 | 1 / 0 / 9 | 4 | — | — |

At this 26 July v26 checkpoint, the restored service remained DCP4/batch-5,120/max-length-524,288 because DCP2 retained one LAVD failure and DCP1's C5 run collapsed. The subsequent issue #33-era campaign above used larger matched samples and additional runtime controls, and superseded that operational selection with DCP2.

### Runtime memory-utilization ceiling

The same three geometries were swept above the original `--gpu-memory-utilization=0.96` baseline. A candidate passed only after engine startup, a near-maximum single-request context with 4,096 generated tokens, and a two-request workload consuming at least 97.8% of the reported aggregate KV budget. The next tested increment had to produce a confirmed CUDA OOM:

| Profile | Highest stress pass | First OOM | KV budget at pass | Gain over 0.96 |
| --- | ---: | ---: | ---: | ---: |
| DCP4 / batch 5,120 | **0.96875** | 0.96900 | 801,792 | +13.7% |
| DCP2 / batch 4,096 | **0.96750** | 0.96775 | 482,944 | +9.4% |
| DCP1 / batch 4,096 | **0.98250** | 0.98275 | 322,496 | +23.8% |

Startup success was not sufficient: DCP4 started at `0.9848` with 978,432 KV tokens but died on its first long-context prefill. The exact passing and failing workloads, OOM allocations, per-trial hashes, and 28 raw harness artifacts are indexed by [`memory-ceiling-summary-20260726.json`](results/v26-tuning/memory-ceiling-summary-20260726.json). These were measured workload boundaries rather than recommendations. The later issue #33-era campaign repeated topology-specific stress tests before selecting GMU 0.9675 for DCP2 production.

## NF3 hybrid v20 result

The 25 July follow-up evaluates `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` at immutable revision `68babde27a97a4c980c2494e830dd424975cd5a3` against the exact attached Gilded Gnosis v20 profile. No full-context mutation completed and cleared the predeclared protocol, so the operational DCP4 service remains byte-identical to the 479,744-token control. No throughput gain is claimed for that unchanged profile.

Two lower-capacity performance Pareto profiles use matched 30-second, explicit-temperature-zero decode cells:

| Profile | Max model length | 8K prefill | 64K prefill | 0K C1 decode | 0K C8 decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced DCP4 control | 479,744 | 3,509 tok/s | 3,247 tok/s | 114.5 tok/s | 320.3 tok/s |
| DCP2 performance | 180,000 | 3,708 tok/s | 3,435 tok/s | 122.7 tok/s | **352.1 tok/s** |
| DCP1 performance | 90,000 | 4,093 tok/s | 3,768 tok/s | **132.8 tok/s** | 346.6 tok/s |

The balanced control passed the structured-tool, exact 450,019-token context, 512-token stress, and post-stress health checks. It did **not** pass the corrective temperature-zero reasoning gates: LAVD measured 1 exact / 5 near / 4 fail with four max-token hits, and Estonia measured 3/10 with seven max-token hits. Earlier LAVD 2/5/3 and Estonia 10/10 observations omitted sampling controls and are retained only as server/model-default evidence. DCP1/DCP2 were not separately quality-gated.

See [`HYBRID_STUDY.md`](HYBRID_STUDY.md) for the audited before/after analysis, protocol deviations, rejection matrix, reproduction commands, and claim boundaries. Raw evidence and the machine-readable scorecard are under [`results/hybrid-v20/`](results/hybrid-v20/).

## RC2+EXL3 MTP78 result

The selected profile combines:

- the exact issue #34 RC2 source pins;
- Brandon Music's EXL3 checkpoint and vLLM/Sparkinfer EXL3 work;
- David Young's dual-plan Trellis prefill work;
- a local rebase of that EXL3 integration onto RC2;
- targeted DCP workspace and required-tool grammar corrections;
- an offline EXL3 conversion of the previously BF16 MTP layer 78 using Brandon Music's published reproduction encoder as the base pipeline.

The accuracy-preserving RC2 profile used `TP4/DCP4/MTP3`, 5,120 maximum batched tokens, eight maximum sequences, and a configured 999,424-token model length. A direct smoke request containing **600,019 prompt tokens** completed and returned `CONTEXT_OK`. No 900K validation is claimed: the attempted 900K harness run was clamped to 128K by its parser and is excluded from publication.

### Cold prefill

Exact-token client measurements, one generated token:

| Prompt tokens | RC2+EXL3 merged baseline | BF16 MTP78 quality build | Initial EXL3 MTP78 profile | Optimized production |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 3,224 tok/s | 3,548 tok/s | 3,557 tok/s | **3,650 tok/s** |
| 64K | 1,782 tok/s | 2,075 tok/s | 1,912 tok/s | **2,109 tok/s** |
| 128K | 1,778 tok/s | 1,945 tok/s | 1,857 tok/s | **1,974 tok/s** |

Increasing the scheduler budget from 3,072 to 5,120 batched tokens improved the fresh matched baseline by 2.8% at 8K, 11.5% at 64K, and 6.2% at 128K. Batch 5,632 did not improve the larger contexts; 6,144 and 8,192 failed with CUDA OOM.

### Sustained decode

Aggregate output tokens/second from steady-state windows:

| Input context | Profile | C1 | C2 | C4 | C8 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | RC2+EXL3 merged baseline | 101.7 | 163.7 | 237.5 | 372.0 |
| 0 | BF16 MTP78 quality build | 100.8 | 166.9 | 244.8 | 372.7 |
| 0 | Initial EXL3 MTP78 profile | 99.5 | 160.2 | 249.5 | 369.6 |
| 0 | Optimized production | 103.6 | **168.6** | 246.9 | **380.7** |
| 128K | RC2+EXL3 merged baseline | 102.4 | 159.7 | capacity-limited | capacity-limited |
| 128K | BF16 MTP78 quality build | 100.8 | 160.2 | capacity-limited | capacity-limited |
| 128K | Initial EXL3 MTP78 profile | **105.2** | 164.3 | 235.9 | capacity-limited |
| 128K | Optimized production | 101.6 | **164.0** | **241.9** | capacity-limited |

Against the fresh 3,072-token control, optimized production improved zero-context C8 by 3.5% and 128K C4 by 2.8%. The repeated 128K C1 cell measured 102.5 tok/s; small single-stream differences were treated as run-to-run variation rather than a kernel gain.

### Quality and behavior gates

LAVD reports exact / near / fail counts over ten deterministic runs:

| Profile | LAVD | Estonia | Required-tool gate |
| --- | ---: | ---: | --- |
| RC2+EXL3 merged baseline | 3 / 5 / 2 | not rerun | not rerun |
| BF16 MTP78 quality build | 4 / 5 / 1 | 10/10 | passed |
| Initial EXL3 MTP78 profile | 6 / 2 / 2 | 10/10 | passed |
| Optimized production | **6 / 3 / 1** | **10/10** | passed |

The faster DCP2 candidate reached 2,976 tok/s at 64K prefill, 2,877 tok/s at 128K prefill, and 422.8 tok/s at zero-context C8, but scored 4 exact / 5 near / 1 fail on LAVD. It missed the predeclared minimum of six exact answers and was not deployed. The DCP4/batch-5,120 profile passed every accuracy, tool, context-capacity, and stability gate.

The uncorrected required-tool path emitted 24 calls—23 duplicate Paris calls followed by `{}`. The corrected grammar emitted exactly one valid call and continued normally. This repository describes that as a targeted structured-output correction, not a general tool-calling result.

### MTP layer 78 conversion

The layer-78 conversion replaced 768 BF16 routed-expert projections with 12,288 EXL3 tensors covering all 256 experts and four TP ranks. The capture retained 131,072 finite rows. The assembled checkpoint reduced stored payload by **15,662,567,424 bytes total**, approximately 3.65 GiB per TP rank.

The conversion pipeline is derivative engineering around Brandon Music's published GLM-5.2 EXL3 reproduction bundle and ExLlamaV3/Trellis machinery. This study does not claim authorship of LDLQ, Trellis, MCG, EXL3, or the original checkpoint encoder.

### Upstream MTP78 and vision controls

The later provenance-hardened MTP78 graft retained 3,904 local KV blocks, passed a 600,019-token text smoke, matched target-only first-token IDs and top-20 log probabilities exactly, reclaimed 3.877–3.896 GiB per GPU, and raised matched draft-token acceptance from 58.21% with BF16 MTP78 to 59.30% with Trellis. It passed a three-run LAVD gate at 2 exact / 1 near / 0 fail and a temperature-zero Estonia gate at 3/3. Long free-form generations were not byte-identical and are not represented as such.

The subsequent GLM-5.2-Vision canary found the same 5/6 semantic result in BF16/Trellis and MTP0/MTP3 modes. The shared B/A image-order failure was already present with MTP disabled. Trellis improved matched vision draft-token acceptance from 72.89% to 77.50%; the separate 200K gate failed with MTP disabled and with text-only input, isolating that capacity defect away from the MTP78 overlay. The decision artifacts and all raw requests, responses, metrics, and synthetic canary images are indexed under [`results/`](results/).

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
- **Complete original-work ledger:** [CONTRIBUTIONS.md](CONTRIBUTIONS.md) separates source implementation, integration, independent discoveries, methodology, upstream ownership, and explicit non-claims across the July GLM-5.2 program.

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
CONTRIBUTIONS.md                 exhaustive cross-project original-work ledger
ISSUE33_STUDY.md                  current issue #33-era plain-English report
HYBRID_STUDY.md                  NF3 hybrid v20 optimization and Pareto analysis
FOLLOWUP_STUDY.md                chronological follow-up analysis
METHODOLOGY.md                   original and RC2 benchmark methodology
CREDITS.md                       ownership, contribution, and license ledger
configs/                         public-safe measured serving configurations
configs/Dockerfile.issue33-exl3  pinned derivative image build recipe
patches/                         source patches applied to pinned RC2 trees
tools/mtp78/                     capture and checkpoint-assembly glue
results/raw/                     original and RC2+EXL3 raw measurements
results/followup/                renamed July 23 follow-up evidence
results/issue34/                 issue #34 RC2 comparison evidence
results/hybrid-v20/              NF3 hybrid gates, scorecard, and raw evidence
results/issue33-upgrade/          86 raw temperature-one issue #33-era artifacts
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
