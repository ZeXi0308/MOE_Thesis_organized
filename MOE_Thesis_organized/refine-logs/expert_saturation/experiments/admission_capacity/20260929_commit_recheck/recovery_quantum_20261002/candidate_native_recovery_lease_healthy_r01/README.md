# Native recovery lease healthy-task adapter r01

CPU_READY / GPU_UNRUN. Independent copy of finalized `candidate_native_recovery_lease_r01`; its manifest `582bd625...acc72` and all original payloads remain unchanged. New manifest: `cf7fdee22ac327ec54ef0d02135bccc65bd17ea2e7cf99bea2d01fb19fc8e0b9` (27 payloads).

Default inputs are 16 GSM8K tasks, burst arrival, natural EOS with 512 maximum output tokens, and OLMoE-Instruct revision `7f1c97f440f06ce36705e4f2b843edb5925f4498`. They were genuinely retokenized using the pinned tokenizer; all32 source prompt texts/token hashes were checked. Separate short/long warmups use the same model. Reference answers and evaluator remain in [healthy_workload](../healthy_workload/README.md).

Only three code files changed: the runner reads task count/output cap/KV bytes from inputs; safe-static accepts the new complete-prompt bounds; measurement retains both `finish_reason` and the real `native_stop_reason`, with explicit empty extra-stop lists. The seven input/stat/warmup files were replaced. All lease, victim, rotation, native-store, offload, memory and policy code is byte-identical to the finalized prototype. See [PROVENANCE.json](PROVENANCE.json).

The default requests **768 usable GPU blocks plus one null block** (`1612709888` bytes). A low-pressure input can request4096 usable blocks. Runtime checks require the actual requested GPU pool/unique tensor bytes, initial-history reservation, 16GiB unique host allocation and8192 host blocks. Engine settings remain max_num_seqs32, max_model_len4096 and token budget1024; the computed safe_cap is descriptive and does not replace the measured cap32.

The existing four-argument shell uses bundled default inputs:

```sh
bash pkg/run.sh native_full_ordinary_only performance ordinary /absolute/output/path
```

Its existing resource/shared-lock environment contract is unchanged. `A_NATIVE_OLDEST_ADMISSION=queue_fund`, `A_NATIVE_OLDEST_REPEAT=1`, and `A_RECOVERY_LEASE_MODE=adaptive|fixed4|q1` select the existing modes. Native/ordinary controls use lease `off` and their original oldest/repeat settings. No launch was performed.

The Python runner retains `--inputs` and `--warmup-inputs`, so an outer coordinated execution runner can pass `healthy_workload/prepared/dev16_burst_kv4096/{inputs,warmups}` or `holdout16_steady_kv768/{inputs,warmups}` without editing measurement code or default payloads. The shell itself intentionally keeps the default package-relative paths. Outer runs must retain the selected workload/config identities and full resource budget.

Validation: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_healthy_runner_cpu.py -v` passes4 focused CPU tests: exact loader and contracts for all three prepared inputs; actual constructor statements for768/4096 with full32; actual safe-cap/live-allocation predicates including rejection of wrong GPU bytes/host block counts; and actual capture code retaining EOS identifier. These use CPU fixtures, not a GPU/model simulation. `python3 verify_package.py` passes27/27. No SSH/GPU/network action occurred.
