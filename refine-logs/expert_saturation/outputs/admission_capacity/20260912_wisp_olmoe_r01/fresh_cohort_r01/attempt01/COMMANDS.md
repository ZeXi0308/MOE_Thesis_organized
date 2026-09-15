# Fresh-cohort F/X repeat

Run only after the previous GPU group releases the shared lock and live GPU checks pass.

```sh
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python run_attempt.py --dry-run
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python run_attempt.py
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python analyze_fresh_cohort.py --input-dir . --out analysis.json
```

Both cohorts use the unchanged eleven runtime files and the compile-domain r02 implementation. No new CPU/GC host observation; all failed cells and costs retained. The driver holds the shared advisory lock across all eight cells; no automatic retry or waiter.
