# Ordinary U compiler localization

Question: do actual Triton compilation or binary/launcher loading intervals account for the previously localized layer0 host stalls? This diagnostic precedes any further X/F performance claim.

Two fresh engines run the same frozen 16-document, 0.25s arrival workload at U24×16, 1GiB KV, P128/O32 and prefill32. The first starts with an empty private Triton disk cache; the second retains the first process's cache. Both have the same instrumentation and one 128→2 warmup. Other subsystem caches are not declared cold. All startup, compilation, warmup, capture, tracing, IO and shutdown costs remain in process wall. No request/output filtering and no subtraction of long calls.

Input code and documents are reused byte-for-byte from finite_arrival_r01. New code is only the compiler/apply observer, bounded two-process driver and direct event joiner. Same-arm source localization supports descriptive attribution of measured intervals; it does not estimate stable cold-vs-warm serving gains or a population noise floor.

CPU preparation (no CUDA imports):

```sh
python3 run_attempt.py --dry-run
python3 analyze_jit.py --input-dir . --out pre_run_analysis.json
```

After the preceding A d6 whole group has terminated or explicitly released its window, verify current GPU/process state and the frozen input hashes. Then launch this driver using the already installed interpreter:

```sh
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python -u run_attempt.py
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python analyze_jit.py --input-dir . --out analysis.json
```

The driver takes `/root/autodl-tmp/moe-research-gpu.lock` with nonblocking flock for both cells, including initialization and gaps. It also validates UUID/empty GPU before each subprocess. Another holder or query failure aborts, with no retry, wait loop or process termination. It never deletes or reuses another experiment's compiler cache. An advisory lock only coordinates adopting drivers; it does not enforce global isolation against other processes.

Retain the complete input bundle, results, private-cache inventories, execution record and logs. Retain failures and unused second cells. Private compiled files stay on the remote instance; their names/size/hash inventories are retained locally. A new attempt needs a new directory and a newly declared cache initial state, not an overwrite.
