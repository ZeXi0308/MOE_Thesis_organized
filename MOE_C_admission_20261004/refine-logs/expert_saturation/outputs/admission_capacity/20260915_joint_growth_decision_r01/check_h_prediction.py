"""Score only the predeclared arrival cohorts using completed primary analysis."""
import argparse
import json
from pathlib import Path

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--analysis", type=Path, default=here.parent / "20260915_natural_cadence_holdout_r02/execution_weste_26862/analysis.json")
parser.add_argument("--output", type=Path, default=here / "H_r02_prediction.json")
args = parser.parse_args()
analysis = json.loads(args.analysis.read_text())
results = []
for block in (0, 1):
    groups = {}
    for name in ("early", "late"):
        arms = {}
        for arm in ("current", "eager"):
            cell = analysis["cells"][f"block{block}-{arm}"]
            rows = [r for r in cell["requests"]["requests"] if (r["arrival_s"] < 12.8) == (name == "early")]
            defined = [r["max_engine_return_gap_s"] for r in rows if r["max_engine_return_gap_s"] is not None]
            maximum = max(defined) if defined else None
            arms[arm] = dict(request_count=len(rows), defined_gap_count=len(defined),
                undefined_gap_count=len(rows)-len(defined), maximum_gap_s=maximum,
                contains_global_maximum=maximum == cell["requests"]["max_engine_return_gap_s"],
                source_comparable=cell["comparable"], source_errors=cell["errors"])
        valid = all(v["request_count"] == 64 and v["source_comparable"] and not v["source_errors"] for v in arms.values())
        baseline, proposed = arms["current"]["maximum_gap_s"], arms["eager"]["maximum_gap_s"]
        ratio = proposed/baseline if valid and baseline is not None and baseline > 0 and proposed is not None else None
        groups[name] = dict(arms=arms, eager_current_ratio=ratio,
            relative_percent=None if ratio is None else 100*(ratio-1),
            prediction_direction_pass=None if ratio is None else ratio < 1)
    results.append(dict(block=block, cohorts=groups))
late = [r["cohorts"]["late"]["prediction_direction_pass"] for r in results]
status = "SUPPORTED_BY_PREDECLARED_DIRECTION_RULE" if all(x is True for x in late) else (
    "NOT_SUPPORTED" if all(x is False for x in late) else "UNCONFIRMED")
out = dict(source_analysis=str(args.analysis), cohort_rule="early arrival_s<12.8; late arrival_s>=12.8; expected64each",
    status=status, blocks=results, budget_result_reused=analysis["transfer_budget"],
    limits=["Undefined gaps remain undefined, never zero-filled.", "No raw, EOS, host or primary-table reanalysis.",
        "Cohort direction support is distinct from the service-cost budget and statistical stability.",
        "All four global maxima are in the late cohort; these are not extra independent repetitions."])
with args.output.open("x") as stream:
    json.dump(out, stream, indent=2)
    stream.write("\n")
print(status)
for item in results:
    print(item["block"], {k:v["relative_percent"] for k,v in item["cohorts"].items()})
