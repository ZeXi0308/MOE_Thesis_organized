# Host cost source localization

FROZEN / GPU UNRUN. Four fresh X engines, observer off/on/on/off, same16 previously measured documents and exact original finite arrivals/resources. Every engine retains r02 M1..160 dual-GEMM compiler/handle setup and all startup cost. Original11 runtime/source files are unchanged.

Only ordinary run_id=measurement activates the new observation. Warmup passes through. The wrapper patches capture before its local import, the actual runpy planner namespace, runtime.kernel and engine.step, and restores them on success/failure. It reuses the pinned CPU counter and GC observer; no device reads or new CUDA synchronization. On spans have engine ordinal and actual call/layer/context identity. Narrow thread/process sample endpoints are retained inside perf intervals. Native per-step cpu_delta includes before/after observation. All wrapper allocation/recording/JSON/startup cost stays in capture or process time as appropriate.

Source question: which CPU execution, GC overlap or waiting indicators accompany residual prefix time before arrival trajectories diverge? Not a new scheduler or performance GO. Off/on may change actual batch/route/output; each executes its own future. No artificial delays or phase scan. Missing/disabled schedstat is unknown, frequency only a sampled value, and GC overlap is not removable savings. Keep every result and no-reproduction.

CPU checks:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 instrumentation/check_host_cost_cpu.py
python3 run_attempt.py --dry-run
python3 analyze_host_cost.py --input-dir . --out pre_run_analysis.json
```

After archive/input verification and the current whole-group GPU handoff:

```sh
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python -u run_attempt.py
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python analyze_host_cost.py --input-dir . --out analysis.json
```

The group holds the shared nonblocking flock and validates UUID/process occupancy each cell; busy/query failure aborts. No kill, automatic waiting, retries or existing-output overwrite. A completed request cell remains COMPLETE even if coverage/observation qualification fails; group stops and retains it.
