# 定向实验完整性复核

Overall PASS；GPT-5.6-Sol ultra fresh agent，same-family/provisional，无P0/P1。

- gt_provenance: pass — Actual native GPU KV snapshot at call328 versus first completed target load; paired state invariant, not quality GT. pkg/selective_kv_roundtrip_check.py:44-58.
- score_normalization: pass — Raw BF16 bytes in logical block/token order. 3296*16*256*2=27000832 per layer, 16 layers=432013312. pkg/recovery_kv_fingerprint.py:9-33.
- result_existence: pass — Independent analyzer reproduction equals analysis.json. First target load115; later116 excluded. No target compute329-331; first resumed step332 starts3296. analysis.json:2-13; offload-events.json:5382-5414.
- dead_code: pass — Hooks installed before measured capture; recorded1867 calls. pkg/run_probe.py:170-183; pkg/selective_kv_roundtrip_check.py:36-67.
- scope: pass — Only first saved full-block prefix. Excludes10-token computed tail, untouched requests, second load, quality and performance. REPORT.md:7-17.
- eval_type: self_supervised paired-state invariant

证据层级：NATIVE_SERVING_INPROCESS_HOST_CAPTURE / single-GPU diagnostic。仅支持 FIRST_RESTORED_PREFIX_MATCH。本审阅不接受质量、泛化保真或性能主张。审阅输入hash见EXPERIMENT_AUDIT.json；原件未修改，无额外审计扩展。
