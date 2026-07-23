# Credits, Ownership, and Source Ledger

This study depends on substantial upstream model, checkpoint, runtime, kernel, tooling, and review work. The purpose of this file is to make the ownership boundary explicit and durable.

Nothing in this repository transfers ownership of, relicenses, or claims authorship over the referenced third-party components. The repository contains the benchmark campaign's original configuration, methodology, measurements, and publication materials; it does not redistribute model weights, container layers, or third-party source trees.

## Component-level attribution

| Component | Owner or contributor | Contribution used by this study | Primary source |
| --- | --- | --- | --- |
| GLM-5.2 base model | Z.ai / GLM team | Base model architecture, weights, tokenizer, model code, and technical work | [Hugging Face](https://huggingface.co/zai-org/GLM-5.2) · [technical report](https://arxiv.org/abs/2602.15763) |
| GLM-5.2 EXL3 TP4 checkpoint | Brandon Music (`brandonmusic`, GitHub `@brandonmmusic-max`) | Calibration/conversion, TP4 rank slicing, EXL3/Trellis routed experts at 3.0 bpw target, BF16 retention for sensitive components, model card, and runtime packaging/validation | [`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw) |
| vLLM EXL3 integration | Brandon Music (`@brandonmmusic-max`) and contributors recorded in commit history | Native rank-sliced EXL3 loading/execution, dense and routed-MoE support, Trellis dispatch, DCP speculative lifetime fixes, structured-output/tool-call fixes, validation, and release pinning | [local-inference-lab/vLLM PR #139](https://github.com/local-inference-lab/vllm/pull/139) |
| Sparkinfer EXL3 Trellis path | Brandon Music (`@brandonmmusic-max`) and Sparkinfer contributors | Planned EXL3 Trellis API and fused Blackwell routed-MoE path | [local-inference-lab/Sparkinfer PR #49](https://github.com/local-inference-lab/sparkinfer/pull/49) |
| Dual-plan Trellis prefill | David Young (`@davidsyoung`) | Second Trellis plan for prefill batches, block-size control, memory-accounted workspace, A/B validation, and CPU contract tests. Incorporated byte-identically into PR #139 with authorship preserved | [local-inference-lab/vLLM PR #163](https://github.com/local-inference-lab/vllm/pull/163) |
| vLLM | vLLM project and contributors | OpenAI-compatible serving engine, scheduling, tensor/context parallelism, KV cache, speculative decoding, and runtime foundations | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| Gilded Gnosis runtime | Local Inference Lab maintainers and individual contributors in its history | Blackwell-focused vLLM runtime, DCP/MTP/MLA correctness and performance work, PCIe collectives, and v20 canonical base | [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) |
| Sparkinfer | Luke Alonso and Sparkinfer contributors | Blackwell inference kernels and runtime APIs used by the planned Trellis path and B12X stack | [local-inference-lab/sparkinfer](https://github.com/local-inference-lab/sparkinfer) |
| ExLlamaV3 / Trellis | turboderp and ExLlamaV3 contributors; underlying Trellis research/authors as cited upstream | EXL3 quantization format, MCG codebook/Trellis machinery, extension behavior, and reference/parity execution used by this checkpoint/runtime | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) |
| Validated container images | Verdict AI plus the runtime contributors above | Publication of immutable runtime images used for the campaign | [Docker Hub](https://hub.docker.com/r/verdictai/glm52-exl3-sparkinfer) |
| Isolated prefill A/B image | David Young (`@davidsyoung`) | Prebuilt image containing the isolated planned-prefill change used during the v2 campaign stage | [vLLM PR #163](https://github.com/local-inference-lab/vllm/pull/163) |
| Benchmark harness | Local Inference Lab `llm-inference-bench` contributors | Exact-token prefill, sustained-decode, hardware monitoring, capacity, Estonia, and LAVD measurement machinery | [local-inference-lab/llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) |
| CUDA/Blackwell platform | NVIDIA and its software/hardware contributors | RTX PRO 6000 Blackwell hardware, CUDA 13.2, drivers, NCCL, and related platform libraries | [NVIDIA](https://www.nvidia.com/) |
| Campaign and publication | Josh Cartu (`@jcartu`) | Four-GPU hardware and power envelope, experiment direction, candidate sweeps, runtime operation, measured artifacts, selection criteria, final validation, and public study archive | [GitHub](https://github.com/jcartu) |
| Development assistance disclosed upstream | OpenAI Codex and Anthropic Claude Code | Implementation, debugging, and documentation assistance disclosed by the author of vLLM PR #139 and Sparkinfer PR #49; the human author selected the design, reviewed changes, and ran validation | [PR #139 disclosure](https://github.com/local-inference-lab/vllm/pull/139) · [PR #49 disclosure](https://github.com/local-inference-lab/sparkinfer/pull/49) |
| Automated review assistance | CodeRabbit | Automated PR summaries and review/check feedback recorded on upstream pull requests | [CodeRabbit](https://www.coderabbit.ai/) |

## Specific immutable artifacts

### Final v20 image

```text
verdictai/glm52-exl3-sparkinfer:v20-gg6722c1d-si1a88b38-cu132-sm120a@sha256:5294b753a81cbed5c7cecd4ef5acdfd1cc13c96bb9233636a42ab8841a439b01
```

The checkpoint model card identifies this image as pinning:

- Gilded Gnosis v20 vLLM `6722c1d`;
- Sparkinfer `1a88b389a8d14f26dbe4c157965938cfd8f1bf51`;
- CUDA 13.2, PyTorch 2.12, and the associated Blackwell runtime stack;
- the rebased vLLM PR #139 and Sparkinfer PR #49 EXL3 work.

### Historical planned-prefill A/B image

```text
davidyoung/glm52-exl3-sparkinfer:v1-prefill-trellis-plan-20260722@sha256:95b7e715e7aca733c44ee6477b2b2abcbed7bfa2bb06acf17b386463be5c0adb
```

David Young described this as the published base image plus exactly the isolated PR #163 `exl3.py` change. The historical [`configs/Dockerfile.exl3-v2-prefill`](configs/Dockerfile.exl3-v2-prefill) records how it was used during the campaign.

### Benchmark harness pin

```text
repository: https://github.com/local-inference-lab/llm-inference-bench
commit: 86cf05c2f42f4d21b909b6e684424ca1aab89fd5
reported result version: 0.4.29
```

## Authorship boundaries

### Work represented as original to this repository

- Design and execution of the July 23 configuration sweep.
- Public-safe copies of the campaign Compose configuration.
- Comparison tables calculated from the archived raw measurements.
- Selection/rejection rationale for the measured candidates.
- Capacity, quality, tool-call, thermal, and power validation performed during the campaign.
- This methodology, manifest, checksum index, and publication layout.

### Work explicitly not claimed

- GLM-5.2 architecture, training, weights, tokenizer, or model code.
- The EXL3 checkpoint's calibration, quantization, conversion, or rank slicing.
- The EXL3/Trellis vLLM loader or execution backend.
- The Sparkinfer Trellis kernels and API.
- David Young's dual-plan planned-prefill implementation.
- Gilded Gnosis, vLLM, Sparkinfer, ExLlamaV3, CUDA, or benchmark-harness source code.
- The referenced Docker image contents.

The campaign tuned and validated configuration around those components; it did not rewrite their authorship as benchmark work.

## Source and license notes

- The checkpoint model card declares MIT and identifies Z.ai's GLM-5.2 as its base model. Consult both model cards and included license files before using the weights.
- vLLM, Sparkinfer, ExLlamaV3, the benchmark harness, and container contents retain the licenses and notices in their source distributions.
- This repository's [MIT License](LICENSE) applies only to original study material placed here by the study publisher. It does not relicense linked or referenced third-party artifacts.
- Raw JSON files are measurements generated by the cited benchmark harness. They are preserved without editing; field names and embedded methodology text originate from that harness.

## Corrections

Attribution errors should be treated as correctness bugs. Open an issue or pull request with the affected component, the correct person/project, and a primary source. The goal is to preserve authorship rather than merely provide a generic acknowledgements list.
