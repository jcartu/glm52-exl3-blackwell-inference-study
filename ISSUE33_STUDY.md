# GLM-5.2 EXL3 on Four RTX PRO 6000 GPUs: Issue #33-Era Study

## Plain-English summary

This study asks a practical question: **what is the best way to serve the 3.0 bpw GLM-5.2 EXL3 checkpoint on four 96 GB RTX PRO 6000 Blackwell GPUs?**

We tested three ways of dividing long-context attention across the four GPUs, raised memory use until the next step produced a real out-of-memory failure, measured short and long prompt processing, measured generation at one through eight simultaneous requests, and ran two long-context reasoning checks repeatedly. Every benchmark in this campaign used an explicit sampling temperature of **1.0**.

The result is not one universal winner:

- **DCP1** is the fastest choice for one user and has the fastest prompt ingestion, but it has the least shared context capacity.
- **DCP4** is the maximum-context choice and completed a 1,048,576-token request envelope, but it is slower for ordinary workloads.
- **DCP2** is the best overall compromise. It nearly matches the old runtime's average generation throughput, improves 64K and 128K prompt processing, exposes 16.6% more KV-cache capacity, and completed a 512,000-token request envelope. It became the selected production profile.

The single largest optimization was surprisingly simple: increasing the transient full-CKV gather ceiling from 16,384 to 140,000 tokens. That improved 64K/128K prompt processing by 15.8%/12.6% under DCP2 and by 49.1%/50.0% under DCP4.

## Important source-boundary note

This is a measurement of the exact source cut used on 26–27 July 2026, not of every later image mentioned in [RTX 6000 Pro issue #33](https://github.com/local-inference-lab/rtx6kpro/issues/33).

The local study image combined:

- issue #33 candidate base image `voipmonitor/vllm:gilded-gnosis-v20-vllm0c79e41-sie603f74-fi801d57a-cu132-20260726` at registry digest `sha256:10261c7d65101c8aba2ce1fb59eabe73aff9d35eca5043b330cc0ce76d3c98d0`;
- vLLM integration `0c79e41db41f250ccdfc4be92d171960a5787f73`;
- Sparkinfer integration `e603f74bb67d0fce547336f1fb73c3c23e8f1887`;
- vLLM EXL3 PR #139 head `26c4bfdd3ff2be0433e6fe07e0c3be535f5bb318`;
- Sparkinfer EXL3 PR #49 head `d4438d490691f79022fdfc8149e1c5f161d15445`;
- FlashInfer `801d57a08958c13d375ddbb6be3be4808f48a708`.

The resulting local image was `local/glm52-exl3-issue33:0c79e41-e603f74-pr139-pr49`, local image ID `sha256:d55205e3ae3d81f00a2770dee91c2bf1662a5efe29c6c897be5ac3010ca75895`.

Issue #33 was subsequently updated with a newer `sic3828fd`/`r4` image and startup-calibration corrections. Those later changes were **not** tested here, and this report does not transfer its numbers to them.

## A few terms, without the jargon

- **Prefill** is the work required to read the prompt before generation begins. Higher prefill tokens/second means a shorter wait before the first generated token for a long prompt.
- **Decode** is generation after the prompt has been processed. Decode tokens/second is the main generation-throughput number.
- **KV cache** is GPU memory used to remember the conversation while generating. More KV capacity permits longer contexts or more simultaneous requests.
- **DCP** means decode-context parallelism. DCP1, DCP2, and DCP4 use one, two, or four context-parallel shards inside the same TP4 model service.
- **MTP3** asks the model's draft head to speculate three future tokens. Accepted draft tokens reduce generation work.
- **C1/C2/C4/C8** mean one, two, four, or eight concurrent requests.
- **GMU** is vLLM's GPU-memory-utilization setting. A service starting successfully at a high value is not enough; a long request can still trigger an unprofiled allocation and fail.

## Hardware and controlled environment

| Component | Value |
| --- | --- |
| GPUs | 4 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB each |
| Aggregate reported VRAM | 391,548 MiB |
| Per-GPU power limit | 600 W |
| GPU order | `3,1,2,0` on this host |
| Checkpoint | `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` |
| Tensor parallelism | TP4 |
| KV format | `nvfp4_ds_mla` |
| Attention | `B12X_MLA_SPARSE` |
| MoE backend | B12X target model, Triton MTP draft |
| Speculation | MTP3, greedy draft sampling |
| Harness | `llm-inference-bench` v0.4.28 |
| Sampling temperature | **1.0 in all 86 raw benchmark JSON artifacts** |

Model loads and benchmarks were serialized. No benchmark was intentionally overlapped with another model load or benchmark.

## Methodology

### 1. Establish a stock control

The matched old-stock service used:

- image `verdictai/glm52-exl3-sparkinfer:v26-gg-v20final-scopefix-archkey-vllm5517197-sibe0edca-cu132-sm120a@sha256:2bb9e804a283d1da3b7e3425ff87375121285141d0d0a40d3dc09d41bf881a10`;
- DCP2, MTP3, GMU 0.96, batch-token limit 4,096, model length 300,000;
- the same checkpoint, GPU order, host, harness, prompt targets, concurrency matrix, and temperature 1.0.

This control reported 440,448 KV tokens in the final matched run.

### 2. Measure prompt ingestion

Cold standalone prefill used exact tokenizer targeting at approximately:

- 8,192 prompt tokens;
- 65,536 prompt tokens;
- 131,072 prompt tokens.

The primary rate is client-observed prompt tokens divided by time to first token. The harness also retained hardware and optional server-counter diagnostics. Prefix-cache reuse was not presented as cold-prefill work.

### 3. Measure generation

Sustained decode used:

- concurrency 1, 2, 4, and 8;
- prompt contexts 0, 32K, and 128K;
- 20-second measured windows after warmup;
- `max_tokens=8192`, explicit `temperature=1.0`, and exact prompt-token targeting;
- continuous OpenAI stream usage as the aggregate-token source.

Cells that could not fit in the available KV cache were marked capacity-limited rather than assigned a throughput value.

### 4. Establish real memory boundaries

A memory setting counted as safe only after:

1. the engine became healthy;
2. a near-ceiling request completed with 4,096 generated tokens;
3. a concurrent stress workload consumed almost all reported KV capacity where applicable;
4. the next tested GMU increment produced a confirmed CUDA OOM or failed the same stress contract.

This distinction mattered. Several higher settings started successfully and then failed on their first long prefill.

### 5. Check long-context answer quality

Two fixed profiles were used:

- **LAVD:** a structured ledger consistency task. Correct final values are 72 tickets and 46 hours. Answers are scored exact, near, or fail.
- **Estonia:** a long-context retrieval/reasoning task whose correct final answer must identify Estonia.

Final comparisons used concurrency 5, `max_tokens=40000`, explicit temperature 1.0, no prefill scout, 20 LAVD runs, and 30 Estonia runs. Earlier ten-run gates were retained and combined only when the runtime/topology and benchmark contract matched. Two-sided Fisher exact tests were used to check whether observed differences supplied evidence of a quality regression.

### 6. Change one control at a time

The study first fixed topology and memory, then changed one control at a time where possible: GPU order, DMA mode and threshold, owner merge, CKV prefetch, CKV gather range, query split, indexer layout, prefill block size, chunk size, Trellis specialization, stream thresholds, supertile size, and KV-RoPE storage.

A few comparisons necessarily changed coupled controls. They are described as mixed evidence and were not used to claim an isolated gain.

## What we tested

### Serving geometry and memory

- DCP1, DCP2, and DCP4 under TP4.
- GMU sweeps around 0.96 through confirmed stress failures.
- Batch-token limits 3,072, 4,096, and 5,120.
- Exact 440K, 500K, 512K, 520K, 735K, and 1M-class request envelopes.
- GPU order `3,1,2,0` versus natural order.

### DCP and communication controls

- query split and an 8,192-token crossover;
- exact top-k owner merge on/off;
- CKV gather on/off and maximum capacities through 140,000 tokens;
- CKV prefetch depths and workspace budget;
- DCP4 project-before-merge workspace;
- indexer shard layouts 0 and 1;
- KV interleave 1 and 16;
- A2A small-batch crossover and AG/RS large-batch backend;
- lossless versus compressed PCIe DMA and a 24 MiB crossover;
- cross-NUMA and single-channel one-shot behavior.

### Kernel and scheduler controls

- EXL3 Trellis min/max M and blocks 8/16;
- prefill Trellis block sizes 8/16/32/48/64;
- prefill chunk 1 versus 128;
- shared-expert stream thresholds 8/16/256 and disabled controls;
- multi-stream GEMM enabled/disabled;
- paged-index supertiles 16K, 32K, and 65,024;
- W4A16 small-M direct route on/off;
- MLA auto strategy and dynamic split heuristic;
- direct-K, stream-scorer, and fused-indexer routes.

### Capacity versus quality controls

- standard NVFP4 MLA KV records;
- `KV_FP8_ROPE=1` as a higher-capacity candidate;
- output-quality gates after final geometry selection.

MTP3 itself was held fixed in this campaign because the immediately preceding v26 matrix had already compared nearby speculative depths and selected MTP3 as the balanced point.

## Results by topology

| Topology | Highest stress-safe GMU | Representative KV budget | Best use | Important verified boundary |
| --- | ---: | ---: | --- | --- |
| DCP1 / batch 4,096 | 0.9825 | 315,584 | single-stream and fastest prefill | 180K configured matrix; next GMU step failed stress |
| DCP2 / batch 3,072 | 0.9675 | **513,536** | balanced production | **507,904 prompt + 4,096 output = 512,000 tokens** |
| DCP4 / batch 5,120 | 0.9780 | 790,272 | higher concurrent long-context capacity | 735K configured service |
| DCP4 / batch 3,072 record | 0.9770 | context-specific | maximum context | **1,048,576-token total request envelope** |

DCP1 produced the highest observed single-request decode and prompt rates. DCP4 could admit a 128K C4 decode cell that DCP2 could not. DCP2 was faster than DCP4 for most ordinary cells and retained a much larger context envelope than DCP1.

## Final production versus matched stock

All rates below are tokens/second.

| Metric | Matched stock DCP2 | Selected issue33-era DCP2 | Change |
| --- | ---: | ---: | ---: |
| KV capacity | 440,448 | **513,536** | **+16.6%** |
| 8K cold prefill | **3,746** | 3,644 | -2.7% |
| 64K cold prefill | 3,155 | **3,380** | **+7.1%** |
| 128K cold prefill | 3,023 | **3,140** | **+3.9%** |
| Common-cell decode geometric mean | baseline | effectively matched | **-0.15%** |

### Full sustained-decode comparison

| Context | Concurrency | Stock | Selected | Change |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 96.6 | **103.8** | +7.4% |
| 0 | 2 | 173.7 | **174.6** | +0.6% |
| 0 | 4 | 261.7 | **269.4** | +2.9% |
| 0 | 8 | **396.1** | 392.5 | -0.9% |
| 32K | 1 | 102.9 | **104.6** | +1.7% |
| 32K | 2 | **172.2** | 164.9 | -4.3% |
| 32K | 4 | 249.4 | **251.0** | +0.6% |
| 32K | 8 | **375.0** | 368.9 | -1.6% |
| 128K | 1 | **104.1** | 100.4 | -3.6% |
| 128K | 2 | **171.9** | 165.3 | -3.8% |

The honest conclusion is not “everything became faster.” The selected profile traded a few long-context decode points for a substantially larger verified context envelope and faster 64K/128K prompt ingestion while preserving overall generation throughput.

## The largest gain: full-CKV gather range

The initial 16,384-token ceiling meant that 64K and 128K prefills fell back to a slower distributed path. Raising the capacity to 140,000 tokens allowed the exact transient full-CKV path to cover the complete standard prefill matrix.

| Topology | Context | Before | After | Change |
| --- | ---: | ---: | ---: | ---: |
| DCP2 | 64K | 2,899 | 3,357 | **+15.8%** |
| DCP2 | 128K | 2,796 | 3,147 | **+12.6%** |
| DCP4 | 64K | 2,186 | 3,260 | **+49.1%** |
| DCP4 | 128K | 2,075 | 3,112 | **+50.0%** |

The final restored service repeated the DCP2 result at 3,380 tok/s for 64K and 3,140 tok/s for 128K. The workspace reduced KV capacity modestly, which is why the selected production envelope is 512K rather than the 520K record obtained with the smaller gather workspace.

## Quality results

### Exact selected DCP2 profile versus matched old stock

| Test | Selected DCP2 | Matched stock DCP2 | Two-sided Fisher p-value |
| --- | ---: | ---: | ---: |
| LAVD acceptable | 19/20 | 20/20 | 1.0000 |
| Estonia pass | 29/30 | 30/30 | 1.0000 |

The selected profile is compared only with the quality files collected on the matched old-stock runtime. The immediately preceding DCP2 final variant, before the 140K CKV-gather ceiling was enabled, independently produced 20/20 LAVD and 29/30 Estonia. It remains in the archive but is not pooled into the primary matched comparison.

### DCP4 descriptive evidence

| DCP4 evidence | LAVD acceptable | Estonia pass |
| --- | ---: | ---: |
| Two final owner-zero variants, pooled descriptively | 38/40 | 56/60 |
| Exact 1,048,576-token envelope profile | not run | 9/10 |

The archive does not contain an old-stock DCP4 quality control, so no DCP4 stock comparison or Fisher claim is made. Early and tuning DCP4 samples used different upgraded configurations and are not relabeled as stock controls.

There was **no statistically detectable difference** in the exact DCP2 matched comparison. This is not proof that the distributions are identical. Estonia's selected-profile point estimate was below 100%, so the study keeps the failure and confidence limitation visible rather than declaring perfect accuracy.

## What we learned from the control audit

### Kept in production

- **MTP3 with greedy draft sampling:** retained from the preceding matched sweep.
- **Dynamic MLA split heuristic (`0`) and auto prefill strategy:** the runtime chooses per shape; a single fixed split could not dominate the full matrix.
- **MLA MG prefill enabled:** this is the supported GLM prefill dispatcher. Disabling it is a hard failure, not a useful fallback.
- **W4A16 small-M direct path:** the off trial was mixed, while source inspection identifies the direct route as the production specialization.
- **Direct-K, stream scorer, and fused indexer enabled:** these are shape-aware fast paths; their off switches are diagnostic fallbacks.
- **DCP2 query split above 8,192 tokens:** a modest long-prefill benefit without forcing the path on tiny contexts.
- **Owner merge off and CKV prefetch depth zero:** both opt-in paths were mixed or slower on this four-GPU placement.
- **Indexer shards 0 and interleave 1:** this preserves capacity and avoids a startup incompatibility.
- **Lossless PCIe DMA above 24 MiB, cross-NUMA allowed, single-channel off:** the measured balance for this host.
- **Trellis block 8, prefill block 64, chunk 1:** valid compiled specializations with the best DCP2 balance.
- **Shared-expert threshold 16 and multi-stream threshold 1,024.**
- **32,768-token indexer supertile:** larger and smaller candidates did not deliver a robust service-level gain.
- **Standard KV RoPE representation:** quality took priority over the additional compressed capacity.

### Rejected or bounded

- **KV FP8 RoPE:** increased capacity by about 15.85%, but the first Estonia gate scored 8/10. It was not deployed.
- **CKV prefetch:** added workspace and repeatedly reduced throughput on this topology.
- **Owner merge:** exact but not a consistent win here. DCP4's owner-zero project-before-merge route was substantially faster.
- **Compressed PCIe DMA:** improved some prefill cells but hurt decode; lossless DMA above the measured threshold was retained.
- **Single-channel one-shot PCIe:** unsafe when target and draft graphs use the transport concurrently.
- **Natural GPU order:** slower than `3,1,2,0` on this host. The chosen order is machine-specific, not a universal recommendation.
- **Trellis block 16:** the required compiled specialization was absent.
- **Supertile 65,024:** changed memory accounting but was performance-neutral or mixed in the service matrix.
- **KV interleave 16:** several C1 cells improved, but other cells regressed; the result was not strong enough to replace interleave 1.
- **Replicated indexer cache:** cost 38,784 KV tokens, about 7.1%, for mixed throughput.

## Known issue found during the study

DCP2 with `VLLM_DCP_INDEXER_SHARDS=1` and query split enabled failed during CUDA-graph profiling:

```text
No indexer query-split group matches the requested KV shard count:
requested=1, partial=None, configured=2
```

Turning query split off allowed the service to start, but the replicated layout consumed 38,784 additional KV tokens and produced only mixed gains. Production therefore uses indexer shards 0. This report records the incompatibility; it does not claim an upstream fix.

## Why DCP2 was selected

DCP2 won the operational decision because it combines:

1. nearly identical geometric-mean decode throughput to stock across ten common cells;
2. faster 64K and 128K prompt processing;
3. 513,536 KV tokens, 16.6% more than the matched stock service;
4. an exact 512,000-token prompt-plus-output completion;
5. stronger observed quality counts than the final DCP4 candidate;
6. materially better ordinary-load throughput than DCP4.

DCP4 remains a valid maximum-context profile. It should be chosen when admitting a 1M-class request or a 128K C4 workload matters more than single-request and moderate-concurrency speed.

## Reproduction

### Build the exact local overlay

The public build recipe is [`configs/Dockerfile.issue33-exl3`](configs/Dockerfile.issue33-exl3). It expects two named source contexts checked out at the exact integration commits represented by the labels:

```bash
docker buildx build \
  --build-context vllm-src=/path/to/vllm-at-0c79e41-plus-pr139-26c4bfdd \
  --build-context sparkinfer-src=/path/to/sparkinfer-at-e603f74-plus-pr49-d4438d49 \
  --file configs/Dockerfile.issue33-exl3 \
  --tag local/glm52-exl3-issue33:0c79e41-e603f74-pr139-pr49 \
  --load \
  .
```

The Dockerfile pins both parent image digests. The resulting local image is derivative runtime packaging and is not represented as an upstream-published image.

### Start the selected service

```bash
export EXL3_MODEL_PATH=/absolute/path/to/GLM-5.2-EXL3-TR3-3.0bpw
export CUDA_VISIBLE_DEVICES=3,1,2,0  # measured host only; choose for your topology

docker compose -f configs/docker-compose.exl3-v20.yml up --force-recreate
```

The Compose defaults reproduce the selected DCP2 policy. Do not blindly copy GMU 0.9675 or GPU order to another machine. Start at 0.96, perform a near-ceiling generation, and establish a real stress failure boundary.

### Reproduce the primary matrix

Run from `llm-inference-bench` v0.4.28:

```bash
python llm_decode_bench.py \
  --host localhost --port 5001 \
  --model GLM-5.2-EXL3-TR3-3.0bpw \
  --concurrency 1,2,4,8 --contexts 0,32k,128k \
  --duration 20 --max-tokens 8192 --temperature 1.0 \
  --standalone-prefill --prefill-contexts 8k,64k,128k \
  --prefill-metric client --token-targeting exact \
  --display-mode plain \
  --output production-inference-temp1.json
```

### Reproduce LAVD

```bash
python llm_decode_bench.py \
  --host localhost --port 5001 \
  --model GLM-5.2-EXL3-TR3-3.0bpw \
  --test-profile lavd-test \
  --profile-concurrency 5 --profile-runs 20 \
  --completion-stats-min-results 20 \
  --max-tokens 40000 --temperature 1.0 \
  --completion-stats-temperature 1.0 \
  --completion-stats-no-prefill-scout \
  --display-mode plain \
  --output lavd20-c5-temp1.json
```

### Reproduce Estonia

```bash
python llm_decode_bench.py \
  --host localhost --port 5001 \
  --model GLM-5.2-EXL3-TR3-3.0bpw \
  --test-profile estonia \
  --profile-concurrency 5 --profile-runs 30 \
  --completion-stats-min-results 30 \
  --max-tokens 40000 --temperature 1.0 \
  --completion-stats-temperature 1.0 \
  --completion-stats-no-prefill-scout \
  --display-mode plain \
  --output estonia30-c5-temp1.json
```

## Evidence map

The archive contains **86 unedited raw benchmark JSON files**, 3.48 MiB total, all successfully parsed and all recording temperature 1.0. Start with [`results/issue33-upgrade/README.md`](results/issue33-upgrade/README.md).

Primary artifacts:

- selected production inference: [`production-final-dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-max512k-inference-temp1.json`](results/issue33-upgrade/production-final-dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-max512k-inference-temp1.json);
- matched stock inference: [`stock-final-dcp2-g960-b4096-inference-temp1.json`](results/issue33-upgrade/stock-final-dcp2-g960-b4096-inference-temp1.json);
- DCP2 512K envelope: [`dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-native512k-full4096-temp1.json`](results/issue33-upgrade/dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-native512k-full4096-temp1.json);
- DCP2 final LAVD and Estonia: [`LAVD`](results/issue33-upgrade/final-dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-lavd20-c5-temp1.json), [`Estonia`](results/issue33-upgrade/final-dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-estonia30-c5-temp1.json);
- DCP4 final inference: [`final-dcp4-g978-owner0-chunk1-pf48-sh16-ckv140k-inference-temp1.json`](results/issue33-upgrade/final-dcp4-g978-owner0-chunk1-pf48-sh16-ckv140k-inference-temp1.json);
- DCP4 1M envelope: [`dcp4-g977-owner0-chunk1-b3072-native1m-full4096-temp1.json`](results/issue33-upgrade/dcp4-g977-owner0-chunk1-b3072-native1m-full4096-temp1.json);
- matched stock quality: [`LAVD`](results/issue33-upgrade/stock-final-dcp2-g960-b4096-lavd20-c5-temp1.json), [`Estonia`](results/issue33-upgrade/stock-final-dcp2-g960-b4096-estonia30-c5-temp1.json);
- generated artifact inventory: [`manifest.json`](results/issue33-upgrade/manifest.json).

## Limitations and claim boundaries

- The results apply to one four-GPU workstation, one physical PCIe/NUMA layout, one 3.0 bpw checkpoint, and the exact source cut above.
- This is not a claim about the later issue #33 `r4` image.
- It is not a new model, quantization method, kernel, or general hardware record.
- Tokens/second is not the same as end-user latency; raw artifacts retain TTFT and ITL details.
- A configured context limit is not accepted as a capacity result unless a request at that envelope completed.
- Temperature-one quality is stochastic. “No statistically detectable regression” is not proof of equality.
- Power and temperature values are sampled and can miss short peaks.
- No independent `p2pmark` bandwidth result is claimed.
- The repository does not redistribute model weights, container layers, or upstream source trees.

## Conclusion

For this machine and source cut, the best normal operating point is **TP4/DCP2/MTP3 with a 512K model limit, 3,072 batch-token budget, 140K full-CKV gather capacity, query split above 8K, owner merge off, prefetch off, and lossless PCIe DMA above 24 MiB**.

The upgrade did not create a universal decode-speed win. It produced something more useful: stock-like overall generation throughput, faster long-prompt ingestion, a verified 16.6% KV-capacity increase, and a cleanly documented route to a 1M-class DCP4 profile when capacity matters more than ordinary-load speed.
