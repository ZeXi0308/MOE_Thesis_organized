# Necessary CPU checks only

Status: **PASS / GPU UNRUN**. Repository HEAD `de64dae5ea4fa3c92ec605e9847e817d8e68f3ac`; the shared workspace was already dirty and is preserved. No CURRENT, G artifact, installed runtime or remote resource was changed.

The offline input preparation verified the pinned Arrow, three tokenizer files and model config; reconstructed whole source articles; excluded the frozen224 prior train documents; selected the first128 within the existing256–3072-token eligibility; and stopped at128. No model weights, GPU, downloads, EOS/output/action/performance filters were used. Its source receipts and decisions are in `used_documents.json` and `pkg/inputs/INPUT_STATS.json`.

`check_preparation.py` passed128 unique document/text/token identities, complete untruncated token lengths, old-row disjointness, source ordering,128 arrivals0.2s apart, explicit EOS/cap semantics and768-request budget. It compared19 inherited runtime/input-warmup payloads plus the generic performance helper byte-for-byte with accepted G. It checked only changed Python syntax, shell syntax, six-cell controller/run agreement and the performance-only CLI.

The new budget function passed inclusive3%/5% bounds, mixed failure, no-maximum-gap reduction and incomplete/UNRUN cases. These are contract checks, not scientific outcome data. G's native source/mode, cadence, EOS, sparse-observer and generic natural-request checks are reused, not rerun. No new diagnostic or test matrix.

Reproduction from repository root:

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_cadence_holdout_r01/check_preparation.py
```

The only handwritten additions are the bounded offline input wrapper, small preparation checks, and budget function; runner/controller/analyzer edits change128-input validation, six-cell order, diagnostic reuse and the declared budget. Shared scheduling/save/load/measurement/resource implementations are unchanged. Manifest and package hash are identity receipts, not GPU qualification or research gains.
