# Source Patches

These diffs record the local source changes used for the measured builds. They are published for review and reconstruction; they are not represented as new upstream releases.

| Patch | Base and purpose |
| --- | --- |
| `rc2-exl3-quality-vllm.patch` | Applies the existing EXL3 integration and dual-plan Trellis work to the exact issue #34 RC2 vLLM source line, plus the measured DCP workspace and single-required-tool corrections. |
| `rc2-exl3-quality-sparkinfer.patch` | Applies the existing EXL3 Trellis path to the exact issue #34 RC2 Sparkinfer source line, including the bounded adaptive sparse-indexer fold used by the measured profile. |
| `adaptive-fold-sparkinfer.patch` | Isolated historical adaptive-fold diff from the July 23 follow-up, retained to make that change reviewable independently. |

Authorship of code present in these diffs remains with the upstream projects and individual contributors in commit/PR history. In particular, the EXL3 work is credited to Brandon Music and contributors, dual-plan prefill to David Young, RC2 work to Martin Vit, `@yatesdr`, and other recorded contributors, and the forced-tool grammar foundation to Florian Bernd. See [`../CREDITS.md`](../CREDITS.md) for exact links.

The local integration, narrow corrections, and patch packaging were directed and validated by Josh Cartu with OpenAI Codex assistance. Upstream licenses govern patched source; this repository's MIT license does not relicense upstream code excerpts.

Apply only to the matching source revisions. Inspect each patch before use:

```bash
git apply --check /path/to/patches/rc2-exl3-quality-vllm.patch
git apply /path/to/patches/rc2-exl3-quality-vllm.patch
```

The final runtime image was built locally and is not available from a public registry.
