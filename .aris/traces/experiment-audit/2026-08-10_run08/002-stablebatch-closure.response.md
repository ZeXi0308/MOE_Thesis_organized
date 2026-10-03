Verdict: **internal artifact closure PASS; external authenticity WARN**. No hash mismatch or cross-run contamination found. The copied bundle is not independently attestable as a remote/Git artifact.

Let `A=artifacts/stablebatch_remote_20260810_run01`.

- Manifest closure passes: recomputed SHA-256 and byte size matched all 7 acceptance entries and all 10 pilot entries. Current file sets exactly equal each manifest, excluding `MANIFEST.json`/`RUN_STATUS.json`. Evidence: `A/outputs/acceptance_20260810_run04/MANIFEST.json:4-30`; `A/outputs/single_contribution_20260810_run01/MANIFEST.json:4-42`.
- JSON/row closure passes: 31/31 JSON and 6/6 JSONL files parse. Rows are workloads 16, candidate sweep 1920, selected targets 32, results 32; result indexes are exactly 0–31. Aggregates recomputed from results match summary: 32 local changes, 12 reproducible route targets, 8 victims, 1 token flip, all integrity fields `PASS` (`summary.json:2-24`).
- V2 binding passes. Recomputed hashes:
  - lock V2 `87d7be8e…ddbaf`
  - config `d49ca39d…efa`
  - runner `97df4862…d57`
  - test `8b41dbec…733`

  These match V2 (`configs/FROZEN_PILOT_LOCK_V2.json:7-11`), pilot request (`run_request.json:17-26`), and static bindings (`static_bindings.json:3-26`). Config snapshot is semantically identical to the bundled frozen config. Acceptance04 and pilot static bindings are byte-identical.
- V1→V2 preserves frozen semantics, claim boundary, config/test/input hashes; only runner hash changes. V2 records the post-run03 inference-mode fix (`FROZEN_PILOT_LOCK_V2.json:5-11`), and current runner contains it at `run_single_contribution_pilot.py:579-580`. **However, V1 runner source is absent, so “integrity-only change” cannot be independently diff-verified.**
- Failure isolation passes:
  - acceptance01/02/03: `FAILED`, `scientific_result_eligible=false`; `FAILURE.json` is byte-identical to `RUN_STATUS.json`; no manifest.
  - acceptance04: `COMPLETE_ACCEPTANCE_ONLY`, ineligible, manifest present, no failure.
  - pilot01: `COMPLETE/SUPPORT`, eligible flag true, manifest present, no failure.
  - Run03 used V1 and ended `07:47:39Z`; V2 froze `07:49:00Z`; acceptance04 and pilot then used V2 sequentially in distinct directories/PIDs. Runner refuses directory reuse (`run_single_contribution_pilot.py:1218-1221`).
- GPU/environment claims are internally consistent: frozen RTX 5090 name/UUID/driver, Torch/Transformers, cuBLASLt hash/version, and matmul flags equal captured environment; pre/post-import process lists are empty; final runtime contains one Python PID matching `run_request.pid`; only expected cuBLASLt is mapped. Evidence: config `:6-24`; pilot `environment.json:7-40`, `runtime_final.json:3-19`; enforcement code `:147-231,236-253`.
- Important closure gap: `RUN_STATUS.json` is deliberately excluded from manifest (`run_single_contribution_pilot.py:1156-1165`) and written after the manifest (`:1316-1327`). Thus `scientific_result_eligible` is not hash-protected. The manifest-bound `summary.json` independently seals `COMPLETE/SUPPORT`, mitigating verdict tampering but not eligibility-flag tampering.
- Provenance gaps: `git_head` and `git_status_short` both contain “not a git repository” (`pilot run_request.json:20-21`). The bundle omits the sealed source manifest, model shards, Transformers source, and cuBLASLt binary referenced by `static_bindings.json:12-24`; their hashes cannot be locally recomputed. There is also no top-level manifest sealing root configs/scripts and failed-run history.

Commands/results relied on:

- `shasum -a 256 …` → all root lock/config/runner/test hashes matched.
- Manifest verifier using `jq`, `shasum`, `stat` → **17/17 PASS**.
- `wc -l -c …` → `16/1920/32/32` rows with manifest-matching bytes.
- `jq -e` over all files → `JSON_FILES=31 INVALID=0`; `JSONL_FILES=6 INVALID=0`.
- Cross-file `jq -n --slurpfile …` → all config/lock/environment/runtime binding predicates `true`.
- Sentinel matrix → three isolated failures, one acceptance-only completion, one pilot completion.

No files were edited.
