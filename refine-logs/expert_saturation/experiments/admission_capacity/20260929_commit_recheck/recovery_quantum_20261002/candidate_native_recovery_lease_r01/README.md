# Native recovery lease prototype r01

Status: CPU implementation checks passed; GPU execution UNRUN. Frozen source candidate is unchanged.

This copy extends repeated oldest `queue_fund` recovery. Before funding and again at commit it selects a quantum within actual full-history plus future-growth KV funding, output cap, and current peer interruption headroom. It retains the existing native recovery and first-output contract. Optional later outputs release at the peer age frontier or when reservation state changes. Selective waiting admission leaves the target its KV growth, one execution token, and a running slot; lease-only deferrals return to ordinary waiting before the next selection.

Run the existing four-argument package entry with `A_NATIVE_OLDEST_ADMISSION=queue_fund`, `A_NATIVE_OLDEST_REPEAT=1`, and `A_RECOVERY_LEASE_MODE=adaptive`, `fixed4`, or `q1`:

```sh
bash pkg/run.sh native_full_ordinary_only performance ordinary /absolute/output/path
```

All existing `H1_*` resource and shared-lock environment variables remain required. The three modes share the native resource/admission path. `adaptive` chooses the minimum feasible quantum amortizing estimated fixed episode overhead to at most 0.5 of episode service; `fixed4` requests four outputs within the same frontier; `q1` requests one. Unknown cost starts at Q1. Quantum cap is 16 and peer age limit is max(1 second, target age at commit). These are exploration settings, not tuned paper results.

Cost sampling uses only past intervals. Decode cadence is median of the latest 32 completed host begin-to-begin intervals whose prior scheduled batch consisted entirely of already-running pure decodes. Episode overhead is median of up to 16 prior protection-start-to-first-output host intervals minus one prior decode interval. Protection start precedes native admission; overlapping device copy times are never added. Estimates provide no wall-clock SLO guarantee.

`selective-store.json` contains `oldest_anchor.lease_decision`, `lease_commit_decision`, `protection_start`, `lease_service_sample`, `lease_execution`, `lease_peer_admissions`, and release events. Existing `raw.json` retains all host output receipts and request completions. `lease_decision_cpu_s` measures model decision calls only, not complete controller overhead.

Local check: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -p 'test_*cpu.py'` (26 tests; 9 new lease lifecycle/resource tests). Tests use actual adapter closures and compile the pinned native scheduler AST; they do not execute a GPU model. `python3 verify_package.py` verifies all 27 payload files.
