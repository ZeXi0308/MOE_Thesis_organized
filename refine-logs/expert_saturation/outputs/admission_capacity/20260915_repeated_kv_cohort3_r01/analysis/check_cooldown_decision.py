"""Compare the existing structural screen with the owner's service summary.

Run from repository root. No raw reread, simulator execution, or fitted timing.
"""
import json
from pathlib import Path

base = Path("refine-logs/expert_saturation/outputs/admission_capacity")
model_path = base / "20260915_repeated_kv_service_r01/start_timing_screen/analysis.json"
service_path = base / "20260915_saved_recovery_start_r01/execution_weste_26862/analysis.json"
model = json.loads(model_path.read_text())["results"]
service = json.loads(service_path.read_text())
current = model["current_save_on"]
eager = model["save_on_without_global_cooldown"]
pairs = service["performance_comparisons"]
assert len(pairs) == 2 and all(p["status"] == "COMPLETE" for p in pairs)
fields = ("total_calls", "mean_completion_step", "load_jobs")
result = {
    "question": "Did the model warn against selecting eager solely for fewer calls?",
    "sources": [str(model_path), str(service_path)],
    "evidence": "CROSS_REFERENCE_OF_EXISTING_MODEL_AND_OWNER_ANALYSIS",
    "model": {k: {"current": current[k], "eager": eager[k]} for k in fields},
    "measured_pairs": [
        {"block": p["block"], "eager_relative_to_current_percent": {
            k: p["eager_relative_to_current_percent"][k]
            for k in ("mean_completion_s", "output_token_rate_s", "max_engine_return_gap_s")
        }} for p in pairs
    ],
    "completion_cost_warning_supported": (
        eager["total_calls"] < current["total_calls"]
        and eager["mean_completion_step"] > current["mean_completion_step"]
        and all(p["eager_relative_to_current_percent"]["mean_completion_s"] > 0 for p in pairs)
    ),
    "limits": [
        "Directional decision warning, not a calibrated second-level prediction.",
        "Model suffix gap and full-episode measured gap have different windows; no error ratio computed.",
        "No new runs, independent samples, significance test, or causal transfer-cost decomposition.",
        "Fewer calls alone cannot select the policy; required pause/efficiency tradeoff remains primary-owner decision.",
    ],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
