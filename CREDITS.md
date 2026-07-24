# Credits, Ownership, and Source Ledger

This study depends on substantial upstream model, checkpoint, runtime, kernel, quantization, tooling, review, and diagnostic work. This ledger makes those boundaries explicit. A benchmark operator integrating and measuring upstream components does not become the author of those components.

Nothing in this repository transfers ownership of, relicenses, or claims authorship over referenced third-party work. Model weights, checkpoint payloads, container layers, and calibration data are not redistributed.

## Component-level attribution

| Component | Owner or contributor | Contribution used by this study | Primary source |
| --- | --- | --- | --- |
| GLM-5.2 base model | Z.ai / GLM team | Architecture, weights, tokenizer, model code, and technical report | [model](https://huggingface.co/zai-org/GLM-5.2) · [report](https://arxiv.org/abs/2602.15763) |
| GLM-5.2 EXL3 TP4 checkpoint | Brandon Music (`brandonmusic`, `@brandonmmusic-max`) | Calibration and conversion, TP4 rank slicing, 3.0 bpw EXL3/Trellis routed experts, sensitive-component retention, model card, validation, and reproduction bundle | [checkpoint](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw) · [encoder bundle](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw/tree/main/encoder) |
| vLLM EXL3 integration | Brandon Music and contributors preserved in commit history | Native rank-sliced EXL3 loading/execution, routed-MoE support, Trellis dispatch, DCP speculative lifetime fixes, structured-output fixes, and validation | [vLLM PR #139](https://github.com/local-inference-lab/vllm/pull/139) |
| Sparkinfer EXL3 Trellis path | Brandon Music and Sparkinfer contributors | Planned EXL3 Trellis API and fused Blackwell routed-MoE path | [Sparkinfer PR #49](https://github.com/local-inference-lab/sparkinfer/pull/49) |
| Dual-plan Trellis prefill | David Young (`@davidsyoung`) | Second Trellis plan for prefill, block-size control, accounted workspace, tests, and A/B validation; incorporated into PR #139 with authorship preserved | [vLLM PR #163](https://github.com/local-inference-lab/vllm/pull/163) |
| Issue #34 RC2 publication and runtime work | Martin Vit (`@voipmonitor`) | Issue #34 test release, image/source pins, DCP/MLA fixes, RC2 integration and related vLLM/Sparkinfer changes | [issue #34](https://github.com/local-inference-lab/rtx6kpro/issues/34) · [vLLM #164](https://github.com/local-inference-lab/vllm/pull/164) · [#167](https://github.com/local-inference-lab/vllm/pull/167) · [#172](https://github.com/local-inference-lab/vllm/pull/172) · [#173](https://github.com/local-inference-lab/vllm/pull/173) · [#174](https://github.com/local-inference-lab/vllm/pull/174) · [Sparkinfer #74](https://github.com/local-inference-lab/sparkinfer/pull/74) · [#75](https://github.com/local-inference-lab/sparkinfer/pull/75) · [#76](https://github.com/local-inference-lab/sparkinfer/pull/76) |
| Additional RC2 vLLM work | [`@yatesdr`](https://github.com/yatesdr) | Contributions incorporated by the issue #34 RC2 source line | [vLLM PR #166](https://github.com/local-inference-lab/vllm/pull/166) · [PR #169](https://github.com/local-inference-lab/vllm/pull/169) |
| Forced-tool and reasoning grammar foundation | Florian Bernd (`@flobernd`) | Earlier forced-tool/reasoning grammar diagnosis and implementation on which this study's narrower repeated-call correction builds | [vLLM PR #34](https://github.com/local-inference-lab/vllm/pull/34) |
| vLLM | vLLM project and contributors | OpenAI-compatible server, scheduling, parallelism, KV cache, speculative decoding, and runtime foundation | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| Gilded Gnosis runtime | Local Inference Lab maintainers and individual contributors in repository history | Blackwell-focused vLLM runtime, DCP/MTP/MLA work, collectives, and RC2 base | [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) |
| Sparkinfer | Luke Alonso and Sparkinfer contributors | Blackwell inference kernels and runtime APIs used by the EXL3/Trellis and B12X paths | [local-inference-lab/sparkinfer](https://github.com/local-inference-lab/sparkinfer) |
| ExLlamaV3 / Trellis | turboderp and ExLlamaV3 contributors; underlying research authors cited upstream | EXL3 format, quantizer operations, MCG/Trellis machinery, extension behavior, and reference execution | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) |
| MTP layer 78 base conversion pipeline | Brandon Music and ExLlamaV3 contributors | Published encoder, calibration schema, LDLQ/Trellis/MCG pipeline, TP slicing, tensor writer/reader helpers, and recipe conventions extended to layer 78 by this study | [checkpoint encoder bundle](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw/tree/main/encoder) |
| Benchmark harness | Local Inference Lab `llm-inference-bench` contributors | Exact-token prefill, sustained decode, hardware monitoring, capacity, Estonia, and LAVD measurement machinery | [repository](https://github.com/local-inference-lab/llm-inference-bench) |
| Historical v20 image publication | Verdict AI and the runtime contributors above | Immutable v20 runtime image used in the July 23 study stage | [Docker Hub](https://hub.docker.com/r/verdictai/glm52-exl3-sparkinfer) |
| CUDA/Blackwell platform | NVIDIA and its hardware/software contributors | RTX PRO 6000 Blackwell hardware, CUDA, drivers, NCCL, and platform libraries | [NVIDIA](https://www.nvidia.com/) |
| Study operation and publication | Josh Cartu (`@jcartu`) | Four-GPU hardware and power envelope, study direction, candidate sweeps, runtime operation, measurements, selection criteria, local integration review, and public archive | [GitHub](https://github.com/jcartu) |
| Local implementation assistance | OpenAI Codex | Assistance implementing, debugging, documenting, and packaging local RC2+EXL3 corrections and MTP78 capture/assembly glue; Josh Cartu directed, reviewed, operated, and validated the work | [`patches/`](patches/) · [`tools/mtp78/`](tools/mtp78/) |
| Development assistance disclosed upstream | OpenAI Codex and Anthropic Claude Code | Assistance disclosed by upstream authors of vLLM PR #139 and Sparkinfer PR #49; human authors retained design, review, and validation responsibility | [PR #139](https://github.com/local-inference-lab/vllm/pull/139) · [PR #49](https://github.com/local-inference-lab/sparkinfer/pull/49) |
| Automated review assistance | CodeRabbit | Automated summaries and review/check feedback visible on upstream pull requests | [CodeRabbit](https://www.coderabbit.ai/) |

## Immutable runtime pins

### Issue #34 RC2 image

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm7e3bee1-si6234185-fi801d57a-cu132-20260723@sha256:67b17855ea81ebc8c9d7fc7c27d0d542c622347cd2607f0cf179e7cc4af2c1f0
```

Recorded source identifiers:

- vLLM `7e3bee1ed4bc...`;
- Sparkinfer `62341856cc54...`;
- FlashInfer `801d57...`;
- issue #34 launcher source `146fa...`.

The final RC2+EXL3 image was a local derivative build. It is not represented as an image published by Martin Vit, Verdict AI, Local Inference Lab, Brandon Music, or any other upstream contributor.

### Historical v20 EXL3 image

```text
verdictai/glm52-exl3-sparkinfer:v20-gg6722c1d-si1a88b38-cu132-sm120a@sha256:5294b753a81cbed5c7cecd4ef5acdfd1cc13c96bb9233636a42ab8841a439b01
```

### Historical planned-prefill A/B image

```text
davidyoung/glm52-exl3-sparkinfer:v1-prefill-trellis-plan-20260722@sha256:95b7e715e7aca733c44ee6477b2b2abcbed7bfa2bb06acf17b386463be5c0adb
```

David Young described this image as the published base plus the isolated PR #163 `exl3.py` change.

### Benchmark harness

```text
repository: https://github.com/local-inference-lab/llm-inference-bench
commit: 86cf05c2f42f4d21b909b6e684424ca1aab89fd5
reported result version: 0.4.29
```

## Original work represented by this repository

- Design and operation of the recorded four-GPU configuration studies.
- Public-safe Compose overlays and publication packaging.
- Measured comparison tables, selection rationale, and explicit rejection boundaries.
- Adaptive sparse-indexer fold selection and final-partial-chunk workspace correction applied to pinned upstream source.
- Rebase/integration packaging of the existing EXL3 PRs onto the exact issue #34 RC2 source line.
- A narrow structural-tag correction for repeated `tool_choice="required"` calls, built on the existing grammar implementation.
- Capture filtering, sealing, resume-driving, validation, and checkpoint-assembly glue used to extend the published encoder pipeline to MTP layer 78.
- Capacity, quality, tool behavior, power, and thermal validation performed on the study host.
- Documentation, manifests, checksum index, and public archive structure.

## Work explicitly not claimed

- GLM-5.2 architecture, training, weights, tokenizer, or model code.
- The original EXL3 checkpoint's calibration, conversion, rank slicing, or model card.
- EXL3, LDLQ, Trellis, MCG, or ExLlamaV3 quantization algorithms and kernels.
- Brandon Music's encoder, vLLM integration, or Sparkinfer Trellis implementation.
- David Young's dual-plan Trellis prefill implementation.
- Martin Vit's, `@yatesdr`'s, or Florian Bernd's upstream runtime and grammar work.
- vLLM, Gilded Gnosis, Sparkinfer, CUDA, FlashInfer, or benchmark-harness source trees.
- The contents or publication of referenced Docker images.
- A public distribution of the locally assembled MTP78 checkpoint or local RC2+EXL3 image.

## Source and license notes

- The checkpoint model card declares MIT and identifies Z.ai GLM-5.2 as its base; consult both model cards and included license files before using weights.
- vLLM, Sparkinfer, ExLlamaV3, FlashInfer, the benchmark harness, container contents, and patch excerpts retain their upstream licenses and notices.
- This repository's [MIT License](LICENSE) applies only to original study material and locally authored helpers where legally separable. It does not relicense linked or patched third-party work.
- Raw JSON measurements are preserved without presentation edits; field names and embedded methodology originate from the cited harness.
- Historical runtime-source copies in `configs/` and source diffs in `patches/` remain subject to their upstream licenses.
- The MTP78 helpers require an independently acquired upstream encoder bundle and checkpoint; neither is copied into this repository.

## Corrections

Attribution errors are correctness bugs. Please open an issue or pull request naming the affected component, the correct person/project, and a primary source so the ledger can be corrected precisely.
