**PASS_CPU_ADAPTER_CONTRACT_ONLY — real vLLM/Qwen GPU initialization remains UNRUN.** `qwen_serial_loader.py` registers `qwen_bf16_serial_v026` as a `BaseModelLoader` subclass and inherits the native `load_model` unchanged. `download_model` only validates local frozen metadata. `load_weights` rechecks that metadata, then calls the existing serial helper once; no default snapshot loader or custom model constructor/postprocessor is used.

Import `qwen_serial_loader` before engine creation. EngineArgs uses this exact configuration, with JSON string paths:

```python
load_format = "qwen_bf16_serial_v026"
model_loader_extra_config = {
    "manifest": "/path/to/qwen3.manifest.json",
    "index": "/path/to/qwen3.index.json",
    "workspace": "/existing/private/staging-parent",
    "receipt": "/existing/output-parent/load.jsonl",
}
```

The model path must be a local config/tokenizer directory whose `config.json` bytes match `manifest.config_sha256`. Revision must be `None` for that local directory or the exact frozen commit; dtype is BF16 with no quantization or `model_weights` override. Workspace/output parents must already exist. Receipt and target-proof paths must be new. The adapter checks vLLM `0.26.0`. Public shard fetches use `urllib.request.urlopen(..., timeout=120)` with no retries; this is a socket-operation timeout, not a full-download or whole-run deadline. The parent runner owns the total deadline.

The installed 0.26 target tracker has a relevant limitation: `track_weights_loading` adds all parameters of a module with postprocessing to its reported loaded set. Therefore this adapter first checks the original returned names against all actual `model.named_parameters()`. It applies exactly the native MoERunner aliases: reported `<runner>.w13_weight/w2_weight` correspond to registered `<runner>.routed_experts.w13_weight/w2_weight`. These aliases come from the actual runner's delegation, `RoutedExperts.load_weights` leaf names and `AutoWeightsLoader` prefixing. The official target tracker then receives a copy, so its additions cannot manufacture the original coverage result.

This produces `<receipt>.targets.json` with `status`, `raw_returned_names`, `raw_names_sha256`, `expected_named_parameters`, `aliases` (reported to registered), `normalized_returned_names`, and `missing`. The digest is SHA256 of `json.dumps(sorted(raw_names), separators=(",", ":")).encode()`. A successful readback requires both the main serial receipt's `COMPLETE` and target proof `PASS` with no missing parameters. Target proof alone does not establish that subsequent cleanup or engine initialization succeeded. Exact HF source-key coverage remains the serial iterator's responsibility; target-container coverage does not independently prove every expert slice was copied.

The [CPU result](cpu_loader_fixture_r01/result.json) came from the [fixture](cpu_loader_fixture.py), executed with local Torch 2.8.0. It extracts and executes the actual 5090 installation's registration function, `BaseModelLoader.load_model`, `_has_online_quant`, and target-tracker method from `installed_sources`; it uses explicit CPU model/platform/import stubs and a serial transport spy. The source SHA256 values are in the result. Observed order was `initialize → serial → model.load_weights → target_coverage → postprocess → eval`. The same native target-check branch that auto-fills postprocessed modules is represented in the fixture: omitting `w2_weight` is rejected before postprocessing, while complete reported aliases pass. A changed index is rejected before loading, and metadata-only validation invokes no transport. The separate real-safetensors iterator fixture remains [cpu_fixture_r01](cpu_fixture_r01/result.json).

The retained first development result is `/private/tmp/qwen3_loader_fixture_r01`; the final alias/timeout fixture was executed at `/private/tmp/qwen3_loader_fixture_r02` and copied without modification to `cpu_loader_fixture_r01`. To rerun, select a fresh output directory:

```sh
.venv/bin/python refine-logs/expert_saturation/outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_streaming_readiness/cpu_loader_fixture.py --output /private/tmp/qwen3_loader_fixture_new
```

This is a CPU adapter qualification, not a successful Qwen model load. There was no large-weight download, GPU access, remote mutation, old-runner change or change to the serial helper's failure-retention behavior. The remaining step is the separately prepared native model-load qualification under measured resource limits.
