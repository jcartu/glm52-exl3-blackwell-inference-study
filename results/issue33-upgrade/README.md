# Issue #33-Era EXL3 Evidence Archive

This directory contains the complete raw benchmark archive behind [`../../ISSUE33_STUDY.md`](../../ISSUE33_STUDY.md).

## Archive contract

- **86 raw JSON benchmark artifacts** were produced between 26 and 27 July 2026.
- Total raw size before this index and derived summaries: **3,647,952 bytes (3.48 MiB)**.
- Every raw JSON file parses successfully.
- Every raw benchmark records `llm-inference-bench` **v0.4.28** and explicit **temperature 1.0**.
- Raw harness outputs are preserved without presentation edits.
- `study-summary.json` is a derived, human-authored decision summary; it is not raw harness output.
- `manifest.json` records byte sizes and SHA-256 values for every file in this directory except itself.

## Read these first

| Question | Artifact |
| --- | --- |
| What did the restored production profile deliver? | [`production-final-dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-max512k-inference-temp1.json`](production-final-dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-max512k-inference-temp1.json) |
| What did matched stock deliver on the same host? | [`stock-final-dcp2-g960-b4096-inference-temp1.json`](stock-final-dcp2-g960-b4096-inference-temp1.json) |
| Did the 512K DCP2 envelope really complete? | [`dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-native512k-full4096-temp1.json`](dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-native512k-full4096-temp1.json) |
| What were final DCP2 quality counts? | [`LAVD`](final-dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-lavd20-c5-temp1.json) · [`Estonia`](final-dcp2-g9675-owner0-chunk1-pf64-sh16-ckv140k-b3072-estonia30-c5-temp1.json) |
| What were matched-stock quality counts? | [`LAVD`](stock-final-dcp2-g960-b4096-lavd20-c5-temp1.json) · [`Estonia`](stock-final-dcp2-g960-b4096-estonia30-c5-temp1.json) |
| What did the final DCP4 candidate deliver? | [`Inference`](final-dcp4-g978-owner0-chunk1-pf48-sh16-ckv140k-inference-temp1.json) · [`LAVD`](final-dcp4-g978-owner0-chunk1-pf48-sh16-ckv140k-lavd20-c5-temp1.json) · [`Estonia`](final-dcp4-g978-owner0-chunk1-pf48-sh16-ckv140k-estonia30-c5-temp1.json) |
| Did a 1M DCP4 envelope really complete? | [`dcp4-g977-owner0-chunk1-b3072-native1m-full4096-temp1.json`](dcp4-g977-owner0-chunk1-b3072-native1m-full4096-temp1.json) |
| What is the machine-readable conclusion? | [`study-summary.json`](study-summary.json) |

## File naming

The filenames intentionally encode the controls that changed:

- `dcp1`, `dcp2`, `dcp4`: context-parallel topology;
- `g9825`, `g9675`, `g978`: GPU-memory utilization, e.g. `g9675` means 0.9675;
- `b3072`, `b4096`: maximum batched tokens;
- `owner0`/`owner1`: exact top-k owner-merge policy;
- `chunk1`: EXL3 prefill chunk;
- `pf48`/`pf64`: prefill Trellis block M;
- `sh16`/`sh256`: shared-expert stream threshold;
- `ckv140k`: transient full-CKV gather capacity;
- `temp1`: explicit temperature 1.0;
- `c5`: profile concurrency five.

Prefixes describe intent:

- `stock-`: final matched old-runtime controls;
- `production-`: restored selected production measurement;
- `final-`: selected topology candidates and quality gates;
- `tune-`: one-control or bounded tuning comparisons;
- `audit-`: explicit flag-audit comparisons;
- `*-native*`, `*-maxctx*`, `*-exact*`, and `*-memory-stress*`: capacity and failure-boundary evidence.

## Major comparison groups

### Initial topology matrix

- `dcp1-g9825-auto-lossless-*`
- `dcp2-g9675-auto-lossless-*`
- `dcp4-g96875-auto-lossless-*`

These establish temperature-one inference, LAVD, Estonia, and memory stress for DCP1/2/4 before the deeper control sweep.

### DCP4 optimization

The sequence covers:

- compressed versus lossless DMA and 24/48/100 MiB thresholds;
- physical GPU order;
- owner merge and project-before-merge workspace;
- chunk size and batch-token budget;
- KV FP8 RoPE capacity/quality;
- exact 1M request envelopes;
- indexer supertile and Trellis block sizes;
- prefill block sizes;
- shared-expert and multi-stream thresholds;
- the final 140K full-CKV gather range.

### DCP2 optimization

The sequence covers:

- owner merge 0/1;
- prefill blocks 32/48/64;
- shared-expert thresholds 16/256;
- chunk 1/128;
- 440K, 500K, 512K, and 520K request envelopes;
- 140K full-CKV gather;
- replicated versus sharded indexer cache;
- query-split coupling;
- KV interleave 1/16;
- final quality and matched stock.

### DCP1 optimization

The sequence covers:

- prefill chunk 1;
- shared-expert thresholds 16/256;
- 65,024-token indexer supertile;
- W4A16 small-M direct route off/on comparison.

## Failed-start observation

The replicated-indexer/query-split combination did not produce a benchmark JSON because the engine never became ready. The observed runtime error was:

```text
No indexer query-split group matches the requested KV shard count:
requested=1, partial=None, configured=2
```

The query-split-off fallback did start and is represented by [`audit-dcp2-indexer-shard1-qsplit0-inference-temp1.json`](audit-dcp2-indexer-shard1-qsplit0-inference-temp1.json). It cost 38,784 KV tokens and produced mixed throughput, so production retained sharded indexer cache (`indexer_shards=0`).

## Integrity

From the repository root:

```bash
sha256sum --check results/SHA256SUMS
```

For this directory alone, inspect [`manifest.json`](manifest.json).

## Privacy and scope

The harness records the study hostname, driver/kernel versions, PCI topology, localhost endpoint, command arguments, sampled hardware state, and local model/cache paths because these affect reproducibility. The archive does not contain model weights, container layers, credentials, API keys, access tokens, private keys, or public service endpoints.
