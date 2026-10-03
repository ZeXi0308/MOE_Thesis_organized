# H1 launch entry: read-only CPU audit

Scope: `candidate_h1/` and `RUN_PLAN.md`, with no SSH, GPU, controller, or package edit. Audited package snapshot: `manifest.json` SHA-256 `04ad1217287491813be3c85839de00ff80ee2ec60011bdf762c58e52b3e3a6d4`; `pkg/run.sh` SHA-256 `40fc323a62983833186a84502fb085e7325536fb6f22d5ab7b29e5005902f7f8`.

## Checked behavior

- `bash -n pkg/run.sh` passed; `PYTHONDONTWRITEBYTECODE=1 python3.13 verify_package.py` verified the exact 25 manifest entries and `pkg/` file set. The manifest SHA check occurs before the verifier under the flock.
- The shell accepts precisely five cells: `native_full_native/performance/off`, `eager/performance/off`, `eager/performance/on`, `eager/diagnostic/off`, and `eager/diagnostic/on`. Only the two `on` cells append `--commit-recheck`; the runner rejects that flag on a non-eager variant. No LTR or second controller is launched by this entry.
- The shell requires an absolute output path, and the updated `realpath -m` guard rejects the candidate package root and descendants, including paths through existing symlinks. The runner creates a fresh output directory with `exist_ok=False`. Outputs should be placed in a separately approved destination; that destination's available capacity and archive path are site gates.
- A `launch-once-${variant}-${mode}-${gate}` directory sits beside `pkg/`; its name is distinct for every allowed cell. It is created **before** package/source preflight, so any failure after this point consumes the cell in that staging. This is the intended fail-closed behavior; recovery needs a separately reviewed staging identity, never an automatic retry.
- Existing fd 9 is accepted only when its inode matches the configured shared lock file; otherwise the shell opens that existing regular file without truncation. `flock -n 9` runs before the manifest/source checks and the measurement. The outer executor must keep its own inherited fd 9 through cell changes and archiving; this script enforces only its own cell scope.
- The selected GPU UUID must occur exactly once in `nvidia-smi --query-gpu=uuid`; `CUDA_VISIBLE_DEVICES` is set to it. The shell requires cgroup v2, compares the supplied `memory.max` path to the current process cgroup's file, and requires a finite value no greater than `H1_APPROVED_HOST_BYTES`. It cannot prove user authorization, cgroup exclusivity, GPU memory headroom, or other jobs' host use; those remain live-site checks.
- `started_s` precedes lock acquisition and all bounded stages. GPU query, package verifier, source preflight, and runner each use the remaining per-cell wall allowance with a 5 s TERM/KILL margin. The site executor still owns the total serial activity budget, transfer, and archive time. The runner's process may be killed before its Python `finally` writes a terminal `status.json`; the shell exit and root ledger must record a timeout as incomplete.
- `HF_HUB_OFFLINE=1` and an existing `HF_HOME` are required. The shell checks `sys.flags.optimize == 0` before `preflight.py` uses assertions to check eight installed vLLM source files. The runner additionally verifies vLLM 0.26.0, two critical source hashes, model/runtime information, physical host KV and GPU KV after engine creation. Actual cached model revision, capacity, source compatibility, EOS, native load completion, and first new output are live-site qualification gates, not CPU-proven results.

## Freeze check

`candidate_h1/README.md`, `provenance.json`, and the actual manifest agree on `04ad1217287491813be3c85839de00ff80ee2ec60011bdf762c58e52b3e3a6d4`. The actual shell hash matches provenance and its manifest entry. The `pkg/run.sh` mode is `0644`; invoke it through `bash` unless the package metadata is intentionally changed and re-reviewed.

## Hardening observations

- `preflight.py` uses `assert` for source SHA checks. The current shell first checks the pinned interpreter's `sys.flags.optimize == 0`, so this entry aborts before preflight if optimization is enabled. An explicit `if`/`raise` inside preflight would also protect against bypassing the shell entry, but this is a defense-in-depth suggestion, not a current launch blocker.
- The current cgroup limit check guarantees a finite cap for that cgroup, but does not check that it contains only this experiment or that an ancestor cap/other processes leave enough usable memory. The executor must verify this on the authorized host.

Status: **CPU launch contract audited, GPU UNRUN**. This report does not qualify the native lifecycle, H1 action, or performance result.
