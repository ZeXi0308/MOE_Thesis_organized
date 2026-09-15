# One numerical qualification process

Local CPU interface check (no torch/vLLM import):

```sh
.venv/bin/python refine-logs/expert_saturation/outputs/admission_capacity/20260912_wisp_olmoe_r01/logical_alignment_qualification_r01/instrumentation/check_logical_alignment_cpu.py
```

After input manifest verification, from the new remote directory:

```sh
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python run_qualification.py --dry-run
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python run_qualification.py
```

The CPU fixture can also read the installed fused_moe.py with `--fused-source`; it checks its frozen SHA and executes only the helper AST with explicit stubs. Its PASS is not a numerical result.

The runner takes the shared nonblocking GPU lock for the entire process, verifies external sources/packages and empty GPU before initialization, and stops on any failure. Its command fixes 5 simultaneous reused requests, P128/O8, token160/prefill32, 384 actual expert slots and 1GiB KV. All reference/prefix/negative overhead is qualification work. No retry in this directory, performance result, or automatic following experiment.
