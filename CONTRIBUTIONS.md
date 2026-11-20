# GLM-5.2 Contribution Ledger

## Purpose and claim standard

This ledger records the original implementation, integration, experimental discovery, methodology, and publication work directed and operated by [Josh Cartu (`@jcartu`)](https://github.com/jcartu) across the public GLM-5.2 program from 5–28 July 2026. The dynamic-LoRA release explicitly records **Sisyphus** as Josh Cartu's implementation identity; the same identity appears in the linked project histories. OpenAI Codex, operated through Oh My Pi, assisted implementation, analysis, review, and documentation.

The word **contribution** is deliberately broader and safer than **invention**:

- **Original implementation** means code first authored for these projects, even when it implements or extends an upstream algorithm.
- **Original integration** means new glue between independently authored systems.
- **Independent discovery** means an effect found and measured in these campaigns. It is not a claim that nobody else could have observed it privately or independently.
- **Campaign methodology** means a test contract designed for these studies. It is not a claim to have invented benchmarking as a field.
- **Upstream work** remains owned and credited to its original authors regardless of how much local integration or validation was required.

Routine service deployment and configuration work performed before the first public original-work commit was reviewed during this audit but is not listed as an invention. The earliest public contribution in scope is the overthinking-penalty implementation published on 5 July 2026.

## Chronology

| Date | Public milestone | Original contribution represented |
| --- | --- | --- |
| 5–6 July | [`vllm-overthinking-penalty`](https://github.com/jcartu/vllm-overthinking-penalty), beginning with [`90cef5b`](https://github.com/jcartu/vllm-overthinking-penalty/commit/90cef5b5f4146507f61ef57f2f12df5c010c4d30) | Speculative-safe vLLM implementation of the paper's marker penalty, runtime controls, experiment harness, corrected GLM-5.2 evidence |
| 16 July | [`glm5.2-unbound` initial release](https://github.com/jcartu/glm5.2-unbound/commit/9837d2502b24ea358247487c5a0107e7e2856e08) | Think-on/off refusal diagnosis, reproducible local serving and verification recipe |
| 21–23 July | [`GLM-5.2 EXL3 Blackwell campaign`](https://github.com/jcartu/glm52-exl3-blackwell-inference-study/commit/27c3f44dae44b8507e43c386b2c9b413466dc2d3) | Controlled EXL3 configuration study, exact-token evidence archive, long-context and quality findings |
| 23–24 July | [`performance/context follow-up`](https://github.com/jcartu/glm52-exl3-blackwell-inference-study/commit/c933f607d68d3a0f89382cf7a636632e5cb1767a) and [`RC2+EXL3 study`](https://github.com/jcartu/glm52-exl3-blackwell-inference-study/commit/d032150376f600236c2ddef8e7713773d2f21ec1) | Adaptive sparse-indexer folding, required-tool correction, RC2 rebase packaging, MTP78 capture/conversion/assembly tooling |
| 23 July | [`Unbound full-method record`](https://github.com/jcartu/glm5.2-unbound/commit/d8902dfa8dedabbe1d3b13d7de8e8aa65e0736a6) | Five checkpoint-surgery variants, activation-steering reference, serving layer, complete retraction and test-methodology record |
| 24–25 July | [`accuracy-preserving EXL3 tuning`](https://github.com/jcartu/glm52-exl3-blackwell-inference-study/commit/c6c20d8ecd0ed01846de7718c9fc482cf79886f9) and [`NF3 hybrid study`](https://github.com/jcartu/glm52-exl3-blackwell-inference-study/commit/138d2fa3496be6ca6189c5f556e75f20cacb40dd) | Gate-bound scheduler tuning, audited DCP Pareto frontier, protocol correction and negative-result publication |
| 26 July | [`v26/MTP78/Vision campaign`](https://github.com/jcartu/glm52-exl3-blackwell-inference-study/commit/dc42487a30760da5817c05ed9dcdfd5f6a95fcf7) and [`memory ceilings`](https://github.com/jcartu/glm52-exl3-blackwell-inference-study/commit/fce3d50c737ede72306c86b3740fb45daf141fe9) | Provenance-hardened MTP78 graft, text/vision probes, topology-aware Pareto matrix, stress-proven memory limits |
| 27 July | [`issue #33-era study`](https://github.com/jcartu/glm52-exl3-blackwell-inference-study/commit/9cc9b55d931970da030e718a7cc898db3c273c56) | Exhaustive flag audit, 140K full-CKV diagnosis, DCP2 512K selection, DCP4 1M envelope |
| 28 July | [`dynamic EXL3 LoRA runtime`](https://github.com/jcartu/glm52-exl3-sparkinfer/releases/tag/exl3-lora-runtime-r2) | Rank-local attention adapters, fully sharded expert adapters, staged adapter-aware Trellis execution, dynamic lifecycle and qualified OCI release |

## Audited prehistory and exclusions

The chronology review began with the first local GLM-5.2 endpoint and OMP configuration work on 23 June, not with the first publication. The 23 June–4 July period consisted of endpoint discovery, model registration, context/compaction configuration, image launches, health checks, checkpoint acquisition, and ordinary benchmark operation. Later work also included routine model swaps, dashboard wiring, cache management, and reproduction of community launch recipes. Those activities were necessary operations, but this ledger does not inflate them into inventions or original research. Where an operational lesson became a new implementation, controlled method, or published finding, it appears below with evidence.

## Original source implementations and integration

### 1. Speculative-safe vLLM overthinking penalty

Repository: [`jcartu/vllm-overthinking-penalty`](https://github.com/jcartu/vllm-overthinking-penalty)

Original project work:

- A stateless server-side vLLM logit-penalty implementation compatible with speculative decoding and KV caching.
- A V2 sampler path and V1 logits-processor fallback.
- A preallocated, vocabulary-sized GPU penalty tensor applied in place rather than rebuilt each decoding step.
- Runtime enable/disable and marker-set controls.
- A parameter-sweep and statistical A/B harness, plus integration with Estonia and LAVD profiles.
- Publication of corrected evidence when an early LAVD interpretation was traced to a truncation/protocol error; see [`2c32f76`](https://github.com/jcartu/vllm-overthinking-penalty/commit/2c32f76f255a1498f9788b7cee5a36ea1395c20f) and [`025c80a`](https://github.com/jcartu/vllm-overthinking-penalty/commit/025c80a21ed8ffd89232a008e03b199c8969cf07).

Boundary: the training-free negative logit penalty and the identification of overthinking markers come from Lotfi, Kirichenko, Li, and Liu, [*Quantized Reasoning Models Think They Need to Think Longer, but They Do Not*](https://arxiv.org/abs/2606.00206). This project implemented and tested that method in vLLM; it did not invent the paper's algorithm.

### 2. GLM-5.2 Unbound implementation and diagnostic method

Repository: [`jcartu/glm5.2-unbound`](https://github.com/jcartu/glm5.2-unbound)

Original project work:

- An independent diagnosis that evaluating an abliterated reasoning model only with thinking enabled can mask a successful weight edit: the chain of thought can re-derive a refusal after the direct answer path has been altered.
- A required evaluation matrix crossing thinking on/off with permissive prompt on/off, plus capability controls.
- Five locally implemented checkpoint-surgery variants, retained rather than collapsed into a success-only narrative:
  1. known-good delta transplant into `o_proj`;
  2. delta transplant plus norm-preserving shared-expert ablation;
  3. stronger self-computed norm-preserving ablation;
  4. whole-residual-footprint ablation;
  5. the selected SRA directional `o_proj` ablation.
- In-place BF16 shard patching with hardlinks/reflinks for untouched shards, so experiments did not duplicate the full checkpoint.
- Reproduction and verification scripts for the selected SRA build and every rejected iteration.
- A reversible activation-steering reference for runtimes with forward-hook support.
- A dedicated client and LiteLLM serving layer that consistently supplies the qualified request controls.
- The explicit retraction of the initial "NF3 experts make this impossible" conclusion once the evaluation-methodology defect was found.

Boundary: refusal-direction research, directional ablation, SRA/NPBA mathematics, community direction tensors, and the evaluated community checkpoints remain credited to Arditi et al., drowzeys, Bahushruth, zandenAI, QuantTrio, AESOP/cfontes, and the other sources named in the [Unbound README](https://github.com/jcartu/glm5.2-unbound#credits). The project contribution is the GLM-5.2 hybrid application, test diagnosis, tooling, controlled comparison, and reproducible serving package.

### 3. Adaptive sparse-indexer fold planning

Evidence: [`patches/adaptive-fold-sparkinfer.patch`](patches/adaptive-fold-sparkinfer.patch) and [FOLLOWUP_STUDY.md](FOLLOWUP_STUDY.md#adaptive-sparse-indexer-folding)

Original project work:

- Replaced a binary two-level-fold switch with an `auto`/forced/off planning policy.
- Calculated candidate-buffer memory before allocation and bounded automatic selection with a configurable MiB ceiling.
- Computed slice counts independently for every chunk, including the shorter final partial chunk, rather than charging every chunk as full width.
- Added validation for invalid policies and boundary tests immediately below and above the configured memory cap.
- Preserved a forced diagnostic mode while making the normal path fail safely back to the unfurled implementation.

Measured outcome: the tested profile exposed 999,424 configured tokens and completed 998,800 prompt tokens plus 32 generated tokens. This does not claim ownership of Sparkinfer's sparse indexer or its original two-level fold.

### 4. Narrow GLM required-tool structural-output correction

Evidence: [`patches/rc2-exl3-quality-vllm.patch`](patches/rc2-exl3-quality-vllm.patch), [FOLLOWUP_STUDY.md](FOLLOWUP_STUDY.md#tool-behavior), and the before/after artifacts indexed in [`results/rc2-exl3-manifest.json`](results/rc2-exl3-manifest.json)

Original project work:

- Added a GLM structural-tag form in which `tool_choice="required"` accepts exactly one tool call rather than an unbounded sequence.
- Preserved the optional reasoning prefix and the existing auto/forced choices.
- Added focused tests for the single-call contract and reasoning-prefix behavior.

Measured outcome: the broken path emitted 24 calls—23 duplicate Paris calls and a final empty object. The corrected path emitted one valid call and continued normally.

Boundary: Florian Bernd's earlier forced-tool/reasoning grammar implementation in [vLLM PR #34](https://github.com/local-inference-lab/vllm/pull/34) is the foundation. The contribution claimed here is only the narrower repeated-required-call correction and its validation.

### 5. MTP layer-78 capture, sealing, assembly, and validation glue

Evidence: [`tools/mtp78/`](tools/mtp78/) and its [README](tools/mtp78/README.md)

Locally authored components:

| Component | Original project role |
| --- | --- |
| `capture/sitecustomize.py` | Capture finite layer-78 hidden rows and routed expert IDs from TP rank 0 under an explicit sentinel |
| `drive_capture.py` | Drive an owner-supplied corpus to a fixed capture target with controlled resume |
| `sanitize_capture.py` | Remove non-finite BF16 rows while preserving alignment with expert IDs |
| `finalize_capture.py` | Enforce shape/routing invariants and seal a fingerprinted capture plan |
| `assemble_checkpoint.py` | Replace only layer-78 BF16 expert projections, verify schemas and hashes, update metadata, emit `MANIFEST.sha256` |
| `assemble_vision_checkpoint.py` | Reflink-clone an immutable text checkpoint and graft pinned MoonViT/projector assets with provenance and tensor inventories |
| `run_text_acceptance_probe.py` | Compare matched MTP acceptance counters |
| `run_vision_canary.py` | Test image order/interleaving while persisting raw HTTP responses before parsing or grading |
| `run_vision_capacity_probe.py` | Exercise long multimodal requests, preserve tokenizer/request/response evidence, and isolate capacity failures |

Measured run: 131,072 finite rows, all 256 experts, 12,288 TP4 EXL3 tensors, and 15,662,567,424 fewer stored bytes than the BF16 layer-78 projections.

Boundary: these tools are derivative glue around Brandon Music's published encoder and ExLlamaV3's LDLQ/Trellis/MCG machinery. They are not a new quantization algorithm or a claim of authorship over the encoder, checkpoint, or tensor format.

### 6. Dynamic, fully sharded EXL3 LoRA serving

Repositories and immutable source commits:

- Runtime and release: [`jcartu/glm52-exl3-sparkinfer`](https://github.com/jcartu/glm52-exl3-sparkinfer), release [`exl3-lora-runtime-r2`](https://github.com/jcartu/glm52-exl3-sparkinfer/releases/tag/exl3-lora-runtime-r2)
- vLLM attention integration: [`d48a599`](https://github.com/jcartu/vllm/commit/d48a59967b95237f3339b7dd81df05237c69b31c)
- vLLM fully sharded experts: [`95d7914`](https://github.com/jcartu/vllm/commit/95d7914de1df93b39fe44957377311ddb752bd2f)
- Sparkinfer staged Trellis execution: [`fc8051e`](https://github.com/jcartu/sparkinfer/commit/fc8051efee755563e2c7a4ce87ce8b683db58381)

Original project work:

#### Rank-local attention adapters

- Safe per-rank reading and validation of a PEFT LoRA manifest.
- Rank-local BF16 factor loading for supported GLM attention targets rather than reconstructing global weights.
- Correct application of the absorbed MLA `kv_b` adapter in projection space through a dedicated kernel path.
- Preservation of unadapted base requests even while an adapter is registered and available.

#### Fully sharded routed-expert adapters

- Recognition of EXL3 `RoutedExperts` as an adapter-capable CUDA-backed layer.
- Mapping of gate/up/down LoRA factors into local expert IDs and tensor-parallel domains.
- A compact adapter-aware execution plan passed from vLLM into Sparkinfer.
- Execution without dequantizing or reconstructing the 332 GB base checkpoint.

#### Adapter-aware Trellis execution

- Public preparation, first-projection, activation, second-projection, and reduction stages.
- Explicit route ownership, expert numbering, aliases, scratch buffers, and CUDA-graph state.
- Bit-for-bit preservation of the no-adapter execution path.

#### Lifecycle and release engineering

- Dynamic HTTP load, unload, and warm reload without restarting the base service.
- Separate base and adapted model IDs from one OpenAI-compatible endpoint.
- Qualified CUDA-graph, mixed base/adapter batch, DCP4 prefix-cache, and MTP-3 paths.
- Digest-pinned image construction from immutable source tags, OCI provenance labels, deployment presets, rollback, and public test evidence.

Qualified evidence includes 109 focused vLLM LoRA tests, 22 CPU MLA regressions, 14 EXL3 bridge/device tests, 29 Sparkinfer GPU tests, bit-for-bit base isolation, 32/32 changed adapted log-probability positions, graph and mixed-batch passes, DCP4 prefix-cache reuse, 86.95% MTP draft acceptance, 18/18 retrieval passes through 30K prompts, tool/streaming gates, and a 30,553-token adapted request. See the release's [README](https://github.com/jcartu/glm52-exl3-sparkinfer#what-was-actually-tested) and [test-suite addendum](https://github.com/jcartu/glm52-exl3-sparkinfer/blob/main/docs/RELEASE_TEST_SUITE.md#2026-07-28-addendum--dynamic-bf16-rank-16-lora-release).

Boundary: vLLM's LoRA framework, GLM-5.2, Brandon Music's EXL3 checkpoint/backend, ExLlamaV3/Trellis, Sparkinfer, and the qualification adapter are upstream or user-supplied work. The release contribution is the missing integration across those layers. The adapter itself is not redistributed and no ownership claim is made over it.

## Independent experimental discoveries and tuning results

### Overthinking-penalty findings

- On the corrected five-run Estonia comparison, the penalty reduced completion tokens by 30.3% with 5/5 correctness in both arms.
- On corrected LAVD, it reduced completion tokens by 23.6%, with 5/5 acceptable results in both arms and a shift from 4 exact + 1 near to 5 exact.
- A separate 20-prompt suite retained 20/20 correctness while reducing hesitation markers by 46.3% and median latency by 24.0%.
- The publication retains the small-sample, reasoning-effort, and model-specific caveats rather than treating these results as universal.

### Unbound findings

- Thinking-on refusal and direct-path refusal are separate test dimensions for a reasoning model.
- Thinking off plus a permissive system prompt exposed successful weight edits that appeared ineffective under thinking-on evaluation.
- The selected abliteration remained load-bearing for the hardest tested categories: think-off alone was not treated as a substitute for weight modification.
- Assistant prefill provided a separately controlled extreme-tail mechanism.
- Capability spot checks remained intact, supporting a narrow "unlock without observed lobotomy" claim rather than a general safety or quality claim.

### Initial EXL3/Trellis campaign

Evidence: [METHODOLOGY.md](METHODOLOGY.md#5-experiment-sequence)

- MTP depth 3 was the balanced speculative point; depths 4 and 5 regressed high-concurrency throughput.
- `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1` materially improved the tested MTP path and was retained without double-counting it in later claims.
- KV-FP8 RoPE was approximately decode-neutral in the initial run and enabled the 786,432-token logical KV geometry.
- A 716,800-token exact input completed; the failed 900K direction was retained as an OOM boundary rather than relabeled as support.
- A 3,072 scheduler budget was balanced; 4,096 reproduced severe low-concurrency regressions in that source cut.
- Triton remained the draft MoE backend because the B12X candidate did not provide a service-level advantage in the matched comparison.
- Absorbed QBMM was incompatible with the checkpoint's rank-sliced EXL3 `kv_b_proj` representation; this was documented as a format/path compatibility result, not a general kernel judgment.
- Against the v2 control, final exact cold prefill improved 58.6% at 8K, 43.8% at 64K, and 37.2% at 128K.
- Raising the LAVD output cap from 16,384 to 24,576 separated a cap-limited response from a model/runtime quality failure.
- Greedy Estonia was not an appropriate quality contract for that configuration; the default sampling policy passed 10/10 while deterministic variants repeatedly hit the cap.

### Adaptive-fold and RC2+EXL3 findings

Evidence: [FOLLOWUP_STUDY.md](FOLLOWUP_STUDY.md)

- The adaptive fold removed a transient candidate-buffer barrier and enabled the 999,424-token configuration described above.
- An experimental DCP2 workspace profile produced much faster prefill and high C8 throughput but failed its deterministic LAVD gate; it was rejected rather than selected by speed alone.
- A local rebase showed that the existing EXL3 and dual-plan Trellis work could operate on the exact issue #34 RC2 source line.
- Offline EXL3 MTP78 conversion removed approximately 3.65 GiB per TP rank while retaining controlled quality, tool, and context gates.
- Raising the scheduler budget from 3,072 to 5,120 improved matched prefill by 2.8%/11.5%/6.2% at 8K/64K/128K; 5,632 did not improve the longer contexts and 6,144/8,192 OOMed.
- The faster DCP2 candidate was rejected for missing the predeclared exact-answer threshold even though it led several throughput cells.

### NF3 hybrid Pareto findings

Evidence: [HYBRID_STUDY.md](HYBRID_STUDY.md)

- No full-context mutation completed every predeclared eligibility and quality gate, so the balanced DCP4 service remained byte-identical to its control and no full-context optimization win was claimed.
- DCP2/180K improved matched zero-context decode by 7.2–10.5% and exact prefill by 5.7–11.0% at the cost of context capacity.
- DCP1/90K improved zero-context C1/C2/C4 by 16.0%/14.1%/21.5% and 8K/64K prefill by 16.6%/16.0%, with substantially less capacity.
- Explicit sampling metadata exposed that an earlier apparent Estonia pass was not a temperature-zero result. Corrective runs failed the declared reasoning gates, and the publication superseded rather than erased the defective evidence.
- A machine-specific GPU rank reorder showed a repeatable 32K/C4 regression and was reverted despite promising prefill.
- MTP4 graph overflow, forced speculative extension, absorb-BMM, FP8 RoPE, oversized scheduler budgets, and 64KB communication thresholds were retained as bounded or negative findings.

### v26 topology and memory findings

Evidence: [README.md](README.md#exl3-v26-temperature-one-pareto-matrix) and [`memory-ceiling-summary-20260726.json`](results/v26-tuning/memory-ceiling-summary-20260726.json)

- DCP1 was the single-stream/prefill point, DCP2 the concurrent-throughput point, and DCP4 the capacity/balanced point for that checkpoint and source cut.
- On this stack, startup alone was insufficient as a memory-safety signal. DCP4 started at GMU 0.9848 with 978,432 KV tokens but died on its first long prefill.
- Stress-proven GMU ceilings were 0.98250 for DCP1, 0.96750 for DCP2, and 0.96875 for DCP4 in the v26 geometry; the next tested increment had to fail the same contract with a confirmed CUDA OOM.
- The stress contract required startup, a near-maximum prompt with 4,096 generated tokens, and concurrent saturation consuming at least 97.8% of the reported KV budget.

### Provenance-hardened MTP78 and vision findings

- The Trellis MTP78 graft reclaimed 3.877–3.896 GiB per GPU, preserved target-only first-token IDs and top-20 log probabilities, and raised matched draft acceptance from 58.21% to 59.30%.
- Long free-form outputs were not byte-identical and were explicitly excluded from identity claims.
- In the matched vision canary, BF16/Trellis and MTP0/MTP3 shared the same 5/6 semantic result. The common B/A image-order failure pre-existed MTP.
- Trellis improved matched vision draft acceptance from 72.89% to 77.50%.
- The separate 200K gate failed with MTP disabled and with text-only input, isolating that capacity defect away from the MTP78 overlay.

### Issue #33-era exhaustive audit

Evidence: [ISSUE33_STUDY.md](ISSUE33_STUDY.md)

#### Full-CKV capacity diagnosis

- The inherited EXL3 recipe's 16,384-token maximum forced 64K and 128K prefill onto a slower distributed path.
- A 140,000-token ceiling covered the complete standard 128K matrix while reserving far less workspace than the source default of 524,288.
- Measured 16,384→140,000 gains were:

| Topology | 64K | 128K |
| --- | ---: | ---: |
| DCP2 | +15.8% | +12.6% |
| DCP4 | +49.1% | +50.0% |

The earliest public exact-140K record found in this audit is study commit [`9cc9b55`](https://github.com/jcartu/glm52-exl3-blackwell-inference-study/commit/9cc9b55d931970da030e718a7cc898db3c273c56), published 27 July at 19:17 UTC. Upstream subsequently reproduced and adopted 140K in [`blackwell-llm-docker` PR #7](https://github.com/local-inference-lab/blackwell-llm-docker/pull/7), created 28 July at 00:44 UTC, and [`rtx6kpro` PR #41](https://github.com/local-inference-lab/rtx6kpro/pull/41), created at 00:56 UTC. No public statement establishes whether that later work was prompted by this study. The defensible claim is the bounded tuning point and measured diagnosis, not ownership of full-CKV gather.

#### Topology and capacity result

- DCP1 was fastest for a single request and prompt ingestion.
- DCP2/batch-3,072 completed 507,904 prompt tokens plus 4,096 output tokens and became the 512K balanced production profile.
- DCP4/batch-3,072 completed an exact 1,048,576-token total request envelope.
- The selected DCP2 profile exposed 16.6% more KV capacity than matched stock while its ten-common-cell decode geometric mean was only 0.15% lower.
- The exact matched DCP2 quality comparison found no statistically detectable difference: LAVD 19/20 versus 20/20 and Estonia 29/30 versus 30/30, both Fisher `p=1.0`. The study did not misstate this as proof of equality.

#### Source-control audit findings

- Query split above 8,192 tokens gave a modest long-prefill benefit without forcing the path on tiny contexts.
- Owner merge and CKV prefetch were exact but mixed or slower on the tested four-GPU placement.
- Lossless PCIe DMA above 24 MiB balanced decode and prefill better than compressed transport.
- Single-channel one-shot PCIe was unsafe when target and draft graphs used the transport concurrently.
- Natural GPU order was slower than `3,1,2,0` on this host; the result was explicitly marked machine-specific.
- KV FP8 RoPE increased capacity by about 15.85% but scored only 8/10 on its first Estonia gate and was rejected.
- Replicated indexer cache consumed 38,784 KV tokens (about 7.1%) for mixed gains.
- DCP2 indexer-shards=1 plus query split exposed an upstream group-selection incompatibility; the study recorded it but did not claim a fix.
- Trellis block 8, prefill block 64, chunk 1, shared-expert threshold 16, multi-stream threshold 1,024, 32,768-token supertile, indexer shards 0, and interleave 1 formed the best measured DCP2 balance.
- Direct-K, stream scorer, fused indexer, W4A16 small-M direct, MLA MG prefill, and dynamic MLA strategy were identified as shape-aware production dispatchers whose disable switches are diagnostic controls, not alternative optimizations.

### Dynamic EXL3 LoRA findings

- Registering an adapter does not perturb base requests: base text and token log probabilities remained bit-for-bit equal across unload.
- Adapter activation changed all 32 compared token-log-probability positions.
- Base-only, adapted-only, and mixed base/adapter CUDA-graph batches completed.
- DCP4 prefix reuse made the repeated base prompt 11.57× faster and the repeated adapted prompt 4.50× faster.
- MTP-3 remained operational with 1,599 of 1,839 proposed draft tokens accepted (86.95%).
- The adapted model was slower on the qualification decode workload (62.76 versus 84.36 tok/s), so no universal adapter speed or quality gain is claimed.

## Campaign-designed methodology and publication infrastructure

The following are original study/release practices and tooling in this program, not claims of global methodological invention:

- Exact prompt-token targeting through the model tokenizer instead of character-count estimates.
- Separate cold-prefill, sustained-decode, capacity, quality, structured-tool, and post-stress health contracts.
- Predeclared gates and supersession records that preserve protocol mistakes and negative candidates.
- A memory acceptance rule requiring a near-ceiling prompt, generated-token headroom, concurrent saturation, and a reproduced OOM boundary.
- Same-host matched controls and geometric-mean comparisons rather than comparing unrelated published headline numbers.
- Fisher exact tests for matched quality counts, with "no detectable difference" kept distinct from proof of equality.
- Synthetic multimodal canaries that vary order, interleaving, MTP, and quantized/BF16 controls.
- Dynamic-LoRA base-isolation checks based on exact text and token log probabilities, not merely HTTP success.
- Immutable source pins, image digests, OCI labels, public-safe Compose overlays, raw unedited JSON, machine-readable manifests, and SHA-256 indexes.
- Claim ledgers that separate configured limits from completed envelopes, cached from cold prefill, sampling policies, capacity-limited cells, and rejected candidates.

The benchmark harness itself is upstream [`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench); these campaigns designed and operated contracts on top of it.

## Integration and publication work

These are substantial contributions but are classified as integration/release engineering rather than new algorithms:

- Rebasing the existing EXL3 and planned Trellis work onto exact Gilded Gnosis v20 and issue #34 RC2 source cuts.
- Building derivative images from pinned source contexts and publishing public-safe reconstruction Dockerfiles.
- Creating topology-specific Compose profiles and explicit rollback paths.
- Capturing and publishing hundreds of raw measurements, hardware summaries, quality responses, manifests, and checksum indexes.
- Maintaining attribution ledgers, source boundaries, rejected-path records, and correction policies.
- Publishing the dynamic LoRA OCI image from immutable source tags with corrected provenance metadata.

## Work explicitly not claimed

This program does **not** claim authorship or ownership of:

- GLM-5.2 architecture, training, weights, tokenizer, chat template, or model code;
- the original GLM-5.2 EXL3 checkpoint, its calibration, TP rank slicing, model card, or reproduction encoder;
- `madeby561`'s MXFP8/NVFP4/NF3 hybrid checkpoint or expert-allocation method;
- the MTP78 head contributed upstream by `malaiwah`;
- the dynamic-LoRA qualification adapter;
- EXL3, LDLQ, Trellis, MCG, ExLlamaV3, or their kernels and formats;
- vLLM's base LoRA framework, scheduler, OpenAI-compatible API, or general model-serving engine;
- Brandon Music's vLLM EXL3 integration or Sparkinfer Trellis implementation;
- David Young's dual-plan Trellis prefill implementation;
- Florian Bernd's forced-tool/reasoning grammar foundation;
- Martin Vit's, `@yatesdr`'s, or other Local Inference Lab runtime work;
- Sparkinfer's sparse indexer, original two-level fold, PCIe collectives, or general MoE kernels;
- full-CKV gather, query split, owner merge, KV compression, speculative decoding, MTP, DCP, or TP as algorithms;
- CUDA, NCCL, FlashInfer, Triton, PEFT/LoRA, or any supporting third-party library;
- referenced model weights, container bases, community refusal directions, or adapters.

Detailed component ownership and license boundaries remain in [CREDITS.md](CREDITS.md) and the dynamic-LoRA release's [CREDITS.md](https://github.com/jcartu/glm52-exl3-sparkinfer/blob/main/CREDITS.md).

## Corrections and additions

This ledger is intended to be exhaustive for the public GLM-5.2 work through 28 July 2026, but attribution errors are correctness bugs. Open an issue or pull request with the affected item, the correct author/project, and a primary source. The claim should be narrowed or corrected before it is defended.
