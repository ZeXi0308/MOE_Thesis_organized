# Independent cohort, fixed reversed order

Input preparation used only pinned local Arrow/tokenizer files and was completed with exclusive outputs:

```sh
.venv/bin/python refine-logs/expert_saturation/outputs/admission_capacity/20260912_wisp_olmoe_r01/logical_alignment_validation_r01/prepare_inputs.py
```

The script reconstructs eligible articles 1..112, including P's prepared inputs, then selects 113..128. Existing prepared/provenance files are never overwritten. Reproduction requires the named local predecessor bundles and pinned dataset/tokenizer cache.

CPU checks from the repository root:

```sh
V=refine-logs/expert_saturation/outputs/admission_capacity/20260912_wisp_olmoe_r01/logical_alignment_validation_r01
.venv/bin/python "$V/run_attempt.py" --dry-run
.venv/bin/python "$V/analyze_logical_performance.py" --input-dir "$V" --out "$V/analysis_unrun.json"
```

After parent authorization and live GPU/lock checks, from an exclusive remote extraction directory:

```sh
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python run_attempt.py --dry-run
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python run_attempt.py
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python analyze_logical_performance.py --input-dir . --out analysis.json
```

The unchanged driver runs `0_Y,1_X,2_F,3_F,4_X,5_Y`. Only Y uses `--logical-alignment`. All arms retain the original compile setup, one first-document 128→2 warmup, 16-request P128/O32 workload, 0.25-second arrivals, 384 expert slots and 1 GiB KV. The driver holds the existing group lock and aborts on occupied GPU, source mismatch, execution or coverage failure; no retry, replacement or process termination is added.
