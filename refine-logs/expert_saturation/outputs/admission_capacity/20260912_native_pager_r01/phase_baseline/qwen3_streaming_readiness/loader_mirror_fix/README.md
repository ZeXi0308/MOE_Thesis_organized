**PASS_CPU_TRANSPORT_MAPPING_ONLY.** Based on the unchanged r02 loader, this [patch](qwen_serial_loader.patch) only adds the frozen mirror fetch callable and passes it to `load_serial`. The new [complete loader](qwen_serial_loader.py) SHA256 is `aad68ed9fd82a75aa82495b0762ecf74e19558b7c4571beac8dcc7eed34a275c`.

Accepted input is exactly `https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/<manifest revision>/<manifest shard basename>`. The callable changes only the hostname to `hf-mirror.com`, preserving the path, and uses `urlopen(timeout=120)`. It has no fallback or retry. Manifest validation occurs when the load_serial argument is evaluated, before that function opens its receipt; the fetch uses the captured manifest and never calls `_metadata()` after the receipt exists. The serial iterator and its whole-shard SHA/size checks, clone/mmap lifecycle and failure retention are unchanged.

[cpu_checks.json](cpu_checks.json) copies the final [CPU result](cpu_r02/result.json). Actual fetch/load_weights AST was executed with explicit metadata, serial and HTTP mocks: all 16 frozen shard URLs mapped correctly, six invalid inputs were rejected without HTTP calls, timeout was 120 seconds, and the serial helper received the intended partial exactly once. The mock creates the receipt before fetching to prove metadata is not revalidated inside fetch. `cpu_r01` retains the earlier URL/wiring fixture result; `cpu_r02` additionally covers this receipt phase boundary. Neither fixture contacts a network or GPU.

Current-machine public prefix reachability is separately recorded by root in `../public_transport_probe_r01.json` and `../mirror_all_shards_probe_r01.json`; prefixes and Content-Length are not full-shard integrity. Complete Qwen transfer remains subject to the serial iterator's exact frozen byte count and SHA256 checks during attempt03. The prior r01/r02 sources and results were not modified.

```sh
.venv/bin/python refine-logs/expert_saturation/outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_streaming_readiness/loader_mirror_fix/cpu_regression.py --output /private/tmp/qwen_mirror_fixture_new
```
