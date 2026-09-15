# Six fixed full-request engines

Local source-order input preparation (already executed; exclusive outputs):

```sh
.venv/bin/python refine-logs/expert_saturation/outputs/admission_capacity/20260912_wisp_olmoe_r01/logical_alignment_performance_r01/prepare_inputs.py
```

After exact input manifest verification, from the new remote directory:

```sh
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python run_attempt.py --dry-run
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python run_attempt.py
```

`run_cells.json` fixes F/X/Y/Y/X/F. Y adds `--logical-alignment` to the compile-covered wrapper, consumes the flag before the original native runner, and runs one logical kernel per actual call. It never enables numerical/reference/negative/prefix qualification work in this performance group.

The runner retains every engine and its private cache, checks the physical GPU before each initialization, and holds the shared lock across the whole group including unload gaps. No GPU waiter or automatic retry. A failed cell remains retained and following cells remain UNRUN.
