# F/X after explicit MoE compile-domain setup

Question: does X retain complete-request advantage over ordinary same-budget F after covering the declared MoE compiler interface domain and retaining full startup cost? This is a descriptive pilot, not a method GO.

F/X/X/F use four fresh engines and the same new16 documents (unused source-pool positions17–32, eligible ordinals49–64). Keep P128/O32,0.25s arrivals,token budget160,prefill32,384 expert slots and1GiB KV. Each process has a separate initially empty Triton disk cache. Other runtime caches are not declared cold.

After original engine initialization, call compile_domain.precompile for M1…160×two GEMMs,320 invocations. F uses E64/global64/mapNone, X uses E384/global384/mapped; shared48 is not X's kernel E. Use original config selection, tensor strides and padding/naive rules. The sole kernel.run call is intercepted with warmup=True, followed by explicit handle initialization. No expert kernel, ensure, weight write or KV operation is executed by this setup. The helper checks that the CPU pager state and tensor pointers remain unchanged, retains temporary bytes and CUDA peaks, and restores the run hook even on failure. It then performs the unchanged one-request128→2 warmup and the finite cohort.

Compile-domain setup is part of process wall. Its timer, temporary memory and native compiler/load events are reported separately. Measurement must contain zero actual compiler_pipeline events to qualify the stated coverage assumption. A miss remains in the raw times, ends this attempt after preserving the completed cell, and is not fixed by deleting the cell or retrying. No claim that every runtime backend is steady or every possible serving shape is covered.

CPU preparation:

```sh
python3 run_attempt.py --dry-run
python3 analyze_covered.py --input-dir . --out pre_run_analysis.json
```

After freezing, uploading and verifying all inputs, read the latest GPU_COORDINATION.md and active registries. Only after the current whole group terminates/releases and a live GPU check passes:

```sh
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python -u run_attempt.py
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python analyze_covered.py --input-dir . --out analysis.json
```

The driver holds common nonblocking flock `/root/autodl-tmp/moe-research-gpu.lock` for all four cells and gaps. A busy GPU, failed query or lock collision aborts. It performs no killing, automatic waiting, retry or old-directory overwrite. Every attempt, compiler-domain JSON, request output, raw trace, observer event, before/after cache inventory, failure and full process time remains retained.
