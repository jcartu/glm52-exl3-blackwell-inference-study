# GLM-5.2 MXFP8/NVFP4/NF3 Hybrid v20 Optimization

## Scope and result

This follow-up evaluates `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` on four NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs against the exact user-supplied Gilded Gnosis v20 service. The sweep targeted exact-token prefill, sustained decode, context capacity, structured output, and reasoning behavior without silently exchanging one property for another.

The result is a three-point Pareto suite, with an important quality caveat:

1. **Balanced operational fallback:** the byte-identical attached TP4/DCP4/MTP3 profile. No full-context mutation completed and cleared the predeclared protocol. This profile preserves the requested 479,744-token service geometry, but the corrective temperature-zero LAVD and Estonia runs failed their predeclared gates; it is therefore not presented as a newly optimized or accuracy-certified winner.
2. **DCP2 / 180K:** the strongest scalable performance-only profile. It improves matched 30-second temperature-zero zero-context decode by 7.2–10.5%, depending on concurrency, while reducing the API limit to 180,000 tokens.
3. **DCP1 / 90K:** the low-concurrency performance profile. It leads DCP2 at C1/C2/C4, while DCP2 is faster at C8; its API limit is 90,000 tokens.

DCP1 and DCP2 were not separately quality-gated. They are capacity/performance Pareto points, not general production recommendations.

## Exact pins

Selected runtime:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm3e731bc-si1a88b38-fi801d57a-cu132-20260722@sha256:dbb3ee5542106442cbdaa21e66d9071fd23888276a2c2cbe914c17491f62ffa7
```

Recorded source revisions:

- vLLM: `3e731bc043d23ec21277fb76d3e15fe6da91b23b`
- Sparkinfer: `1a88b389a8d14f26dbe4c157965938cfd8f1bf51`
- FlashInfer: `801d57a`
- CUDA: 13.2.1
- NCCL observed at runtime: 2.30.4
- benchmark: Local Inference Lab `llm-inference-bench` 0.4.29

Checkpoint:

```text
madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid@68babde27a97a4c980c2494e830dd424975cd5a3
```

[`checkpoint-provenance-20260725.json`](results/hybrid-v20/checkpoint-provenance-20260725.json) records the immutable Hugging Face revision and SHA-256 values for the local config, generation config, tokenizer config, chat template, tokenizer, tensor index, and MXFP8 sidecar. Four small local files were byte-compared with `resolve/<revision>/` and matched.

The newer 23 July RC2 image was tested once at matched DCP4 geometry rather than assumed better. It measured 3,354 / 2,718 / 2,797 tok/s at 8K / 64K / 128K prefill and showed lower long-context decode, so it was not advanced. That exploratory screen was not repeated under the predeclared confirmation rule.

## Predeclared gates and execution audit

The gates were written before candidate selection and remain unchanged in [`hybrid-optimization-gates-20260725.json`](results/hybrid-v20/hybrid-optimization-gates-20260725.json):

- balanced candidates had to retain `MAX_MODEL_LEN=479744`;
- a comparable performance cell below 97% of baseline required confirmation and rejected the candidate if repeated;
- LAVD required ten runs, concurrency five, explicit temperature zero, a 40,000-token cap, no errors, and no max-token hits;
- GSM8K used a deterministic 100-item slice at temperature zero;
- Estonia required 10/10 correct at explicit temperature zero with no errors;
- the structured-output gate required exactly one `get_weather({"city":"Paris"})` call and a normal continuation;
- the capacity gate required an exact 450,019-token prompt plus a 512-token cached decode stress without OOM, Xid, engine death, or loss of health;
- the declared primary score was the geometric mean of comparable exact-prefill and sustained-decode ratios.

The post-run audit found two execution defects in the first evidence set:

1. The first LAVD and Estonia commands omitted `--completion-stats-temperature 0`. Their JSON metadata records `temperature: null` and `top_p: null`; they are server/model-default observations, not temperature-zero evidence. The checkpoint's `generation_config.json` declares temperature 1.0 and top-p 0.95.
2. The first DCP1/DCP2 headline decode artifacts used 20-second cells. A first 30-second response rerun then omitted the explicit decode temperature. Both sets are retained as superseded evidence, but neither drives the published matched comparison.

Corrective runs used `--completion-stats-temperature 0` for completion profiles and `--temperature 0`, 30-second windows, and `ignore_eos` for sustained decode. The full audit and supersession map are machine-readable in [`hybrid-protocol-amendment-20260725.json`](results/hybrid-v20/hybrid-protocol-amendment-20260725.json).

The declared geometric mean was not computed. No mutated full-context candidate completed every declared cell and eligibility gate, and candidate 240,041/400,000-token prefill cells were not collected after earlier screening failures or operational boundaries. Direct uncached exact-token measurements were added for the unchanged selected control, but the study does not retroactively claim that every mutation was ranked by the original geometric mean.
 
The repeat rule was also only partially executed. The rank-order candidate's 32K/C4 regression received both a fresh matrix and a 60-second confirmation; RC2, MTP, RoPE, threshold, scheduler, and several other exploratory regressions were not systematically repeated. Those single-run rows explain why profiles were not advanced in this sweep, not that every mutation is intrinsically or repeatably slower.

## Attached v20 control

### Exact cold prefill

| Prompt tokens | Throughput | Measurement path |
| ---: | ---: | --- |
| 8,192 | 3,509 tok/s | harness exact targeting |
| 65,536 | 3,247 tok/s | harness exact targeting |
| 131,072 | 2,929 tok/s | harness exact targeting |
| 240,041 | 2,880.9 tok/s | direct exact-token streaming client |
| 400,000 | 2,568.9 tok/s | direct exact-token streaming client |

The 240K and 400K requests used unique prefixes, exact `/tokenize` binary search, `max_tokens=1`, temperature zero, and reported `cached_tokens=0`. See [`hybrid-sota-final-prefill-long-exact-20260725.json`](results/hybrid-v20/hybrid-sota-final-prefill-long-exact-20260725.json).

### Sustained aggregate decode

All cells below use 30-second windows, explicit temperature zero, `ignore_eos`, and continuous-usage output tokens/s.

| Input context | C1 | C2 | C4 | C8 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 114.5 | 130.7 | 214.4 | 320.3 |
| 32K | 114.5 | 128.2 | 213.9 | 310.2 |
| 128K | 110.4 | 125.9 | capacity-limited | capacity-limited |
| 300K | 117.5 | capacity-limited | capacity-limited | capacity-limited |

### Corrective quality and behavior results

| Gate | Corrective result | Status |
| --- | --- | --- |
| LAVD, explicit temperature 0 | 1 exact / 5 near / 4 fail; 0 errors; 4 max-token hits | **failed** |
| GSM8K, temperature 0 | 98/100; 0 errors; failures `gsm8k-0962`, `gsm8k-1042` | matched control observation |
| Estonia, explicit temperature 0 | 3/10 correct; 0 errors; 7 max-token hits | **failed** |
| required tool | one valid Paris call, then normal continuation | passed |
| 450,019-token context and stress | uncached `CONTEXT_OK`, then cached 512-token completion | passed |
| post-stress health | model list 200, max length 479,744, service ready, 0 error-pattern matches, 0 recent Xid events | passed |

The earlier omitted-sampling observations were LAVD 2/5/3 and Estonia 10/10. They remain in the raw archive but are not used to claim a temperature-zero pass. The explicit results mean this study does **not** certify the checkpoint/profile against its predeclared temperature-zero reasoning gates.

## Performance/capacity Pareto frontier

### Exact prefill

| Profile | Max model length | 8K | Δ | 64K | Δ | 128K | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Attached DCP4 control | 479,744 | 3,509 | — | 3,247 | — | 2,929 | — |
| DCP2 performance profile | 180,000 | 3,708 | +5.7% | 3,435 | +5.8% | 3,252 | +11.0% |
| DCP1 performance profile | 90,000 | 4,093 | +16.6% | 3,768 | +16.0% | not runnable | — |

### Zero-context sustained decode

| Profile | Max model length | C1 | Δ | C2 | Δ | C4 | Δ | C8 | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Attached DCP4 control | 479,744 | 114.5 | — | 130.7 | — | 214.4 | — | 320.3 | — |
| DCP2 performance profile | 180,000 | 122.7 | +7.2% | 144.4 | +10.5% | 235.4 | +9.8% | **352.1** | +9.9% |
| DCP1 performance profile | 90,000 | **132.8** | +16.0% | **149.1** | +14.1% | **260.5** | +21.5% | 346.6 | +8.2% |

DCP2 exposed a 197,376-token measured KV budget; DCP1 exposed 118,912 tokens. Those capacities are not equivalent to the balanced control's 487,936-token measured KV budget and 479,744-token API limit.

At 32K, DCP2 measured 121.9 / 141.7 / 232.8 tok/s at C1/C2/C4; C8 was capacity-limited. DCP1 measured 132.0 / 143.3 tok/s at C1/C2; C4/C8 were capacity-limited.

## Why the full-context configuration did not change

The sweep covered runtime pin, scheduler budget, GPU rank order, DCP, MTP, speculative-decode behavior, KV RoPE behavior, absorb-BMM behavior, DCP A2A threshold, PCIe all-reduce threshold, graph behavior, and memory-utilization/capacity boundaries.

The most promising full-context mutation reordered ranks to `3,1,2,0`. It measured 3,461 / 3,243 / 3,108 tok/s prefill. Its original quality observations used omitted sampling controls and are not gate-comparable. Independently, a fresh production matrix measured 32K/C4 at 203.1 tok/s and a 60-second confirmation measured 196.0 tok/s, versus 213.9 tok/s baseline. Both violate the 0.97 minimum-cell gate, so production returned to `0,1,2,3`.

A later repeat of the byte-identical control produced 3,512 / 3,244 / 3,109 tok/s prefill, while several decode cells moved beyond the 3% band. MTP acceptance, run order, and thermal state make isolated cells noisy enough that repeat values are published as variance evidence. Because the final configuration is byte-identical to the control, no causal balanced-profile throughput gain is claimed.

Important screening and operational boundaries:

- batch 5,120 retained only 332,544 KV tokens and could not serve 479,744 tokens;
- batch 4,096 at GMU 0.985 estimated only 472,064 tokens; GMU 0.986 could not start;
- batch 3,584 booted but failed prefill with CUDA OOM;
- batch 3,200 reduced headroom without a gate-clearing gain;
- MTP4 collapsed zero-context C8 to 49.7 tok/s after graph overflow;
- MTP2 reduced C1 without a compensating scalable gain;
- explicit speculative-extend-as-decode reduced KV headroom and throughput;
- explicit absorb-BMM over-allocated KV and failed around 135K;
- explicit FP8 KV RoPE reduced 128K prefill to 1,947 tok/s;
- 64KB DCP A2A and PCIe all-reduce thresholds produced mixed or regressive cells.

Startup-only boundaries are explicitly labeled operator observations in [`startup-failure-observations-20260725.json`](results/hybrid-v20/startup-failure-observations-20260725.json). The absorb-BMM harness artifacts are also public. The scorecard records artifact status and known evidence gaps rather than calling every row machine-verifiable: [`hybrid-v20-scorecard-20260725.json`](results/hybrid-v20/hybrid-v20-scorecard-20260725.json).

## Reproduction

Download the immutable checkpoint revision and use that directory as `MODEL_DIR`:

```bash
huggingface-cli download \
  madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid \
  --revision 68babde27a97a4c980c2494e830dd424975cd5a3 \
  --local-dir "$HOME/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid"

export MODEL_DIR="$HOME/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid"
```

Balanced operational profile:

```bash
docker compose \
  -f configs/docker-compose.nf3-v20-balanced.yml \
  up glm52-nf3-v20
```

DCP2 / 180K performance profile:

```bash
docker compose \
  -f configs/docker-compose.nf3-v20-balanced.yml \
  -f configs/docker-compose.nf3-v20-dcp2-180k.yml \
  up glm52-nf3-v20
```

DCP1 / 90K performance profile:

```bash
docker compose \
  -f configs/docker-compose.nf3-v20-balanced.yml \
  -f configs/docker-compose.nf3-v20-dcp1-90k.yml \
  up glm52-nf3-v20
```

The image's launcher overwrites the all-reduce thresholds by default. [`serve-glm52-v16.v20-hybrid.sh`](configs/serve-glm52-v16.v20-hybrid.sh) is the measured image launcher with only those two assignments changed to honor preset environment values. The balanced Compose file mounts it over `/usr/local/bin/serve-glm52-v16.sh`.

GPU rank order is topology-specific. The measured operational default is `0,1,2,3`; override `HYBRID_GPUS` only after repeating the complete performance and behavior protocol on the target machine.

## Claim boundaries

- This is not a new model, quantization format, kernel, or general performance record.
- The unchanged DCP4 configuration is an operational fallback, not a claimed optimization gain or temperature-zero accuracy-certified winner.
- DCP1 and DCP2 results are performance/capacity Pareto points and were not separately quality-gated.
- The predeclared primary geometric mean was not computed; the execution audit explains why.
- Corrective LAVD and Estonia temperature-zero results failed. The earlier omitted-sampling observations are not relabeled as deterministic evidence.
- Raw artifacts include machine topology and local-path metadata needed for reproducibility; they contain no weights, container layers, credentials, or private endpoints.
