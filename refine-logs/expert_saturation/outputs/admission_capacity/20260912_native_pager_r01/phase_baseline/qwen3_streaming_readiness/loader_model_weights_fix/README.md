**PASS_CPU_DEFAULT_REGRESSION_ONLY.** The installed vLLM 0.26 `ModelConfig.model_weights` default is `""` ([actual source](../installed_sources/config/model.py:116)), not the model path. Its local-model branch returns without changing that empty default ([source](../installed_sources/config/model.py:892)). Only the object-storage branch sets a weights URI. This explains r01's rejection before the serial iterator was called.

The [patch](qwen_serial_loader.patch) changes only this guard: normalize the exact empty default to the already validated local model path, then allow `None` or a path resolving to that same directory. A distinct directory is rejected even if its config bytes match; an external URI is rejected. All earlier config/index/revision/BF16 checks, target coverage, HTTP timeout and serial failure retention remain intact.

Use the complete new [loader source](qwen_serial_loader.py), SHA256 `719c5a8b61471a14f70a9ef4be9d9e0f9d0551c09a0f1123cc93fe8be9c74a3c`, for attempt02. The original loader remains SHA256 `ce2282397da395be7adce9a7137bd69b38f18564d2bbbb67cfbc76ae62597988`; r01 source and evidence were not changed.

[cpu_checks.json](cpu_checks.json) is an exact copy of the executed [CPU result](cpu_r01/result.json). The targeted fixture extracts the actual installed default field and local-processing method, reproduces the old rejection, accepts None/empty/same-local, and rejects a different local directory plus an external URI. It executes actual loader metadata validation, with a local-URI predicate fixture; it does not instantiate the full vLLM ModelConfig, fetch weights or access a GPU. The installed config source SHA256 is `7d52e060db54067a21591882bd6e3c2dd2486ac9041dde1117f97023e71bf89f`.

```sh
.venv/bin/python refine-logs/expert_saturation/outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_streaming_readiness/loader_model_weights_fix/cpu_regression.py --output /private/tmp/qwen_model_weights_regression_new
```

Attempt02's actual model load remains unrun by this subtask.
