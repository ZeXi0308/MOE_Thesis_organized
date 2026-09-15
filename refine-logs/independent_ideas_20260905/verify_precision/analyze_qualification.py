"""Read every planned run; never choose favorable repeats or invent denominators."""
import argparse
import json
import math
from pathlib import Path
import statistics

from budget_adapter import POLICIES


def analyze(path):
    config = json.loads((path / "config.json").read_text())
    documents = json.loads((path / "documents.json").read_text())["documents"]
    rows, missing, invalid = [], [], []
    for repeat in range(config["repeats"]):
        for index, document in enumerate(documents):
            arms = {}
            for policy in POLICIES:
                filename = f"r{repeat:02d}-d{index:03d}-{policy}.json"
                if not (path / filename).is_file():
                    missing.append(filename)
                    continue
                try:
                    result = json.loads((path / filename).read_text())
                    ledger = result["summary"]
                    if (result["document_sha256"] != document["document_sha256"] or result["policy"] != policy
                        or result["repeat"] != repeat or [row["step"] for row in result["steps"]] != list(range(32))
                        or sum(row["action"] == "high" for row in result["steps"]) != 4
                        or any(not math.isfinite(row["reference_kl"]) or row["reference_kl"] < 0 for row in result["steps"])
                        or not math.isclose(sum(row["reference_kl"] for row in result["steps"]), ledger["accumulated_kl"], rel_tol=1e-12, abs_tol=1e-12)
                        or (ledger["served_high_steps"], ledger["physical_high_calls"], ledger["physical_low_calls"]) != (4, 32, 32)):
                        raise ValueError("identity or budget/accounting mismatch")
                except (ValueError, KeyError, TypeError):
                    invalid.append(filename)
                    continue
                arms[policy] = ledger
            if len(arms) == 3:
                fixed, reactive, random = (arms[policy] for policy in POLICIES)
                rows.append(dict(document=index, repeat=repeat, fixed_kl=fixed["accumulated_kl"],
                    reactive_kl=reactive["accumulated_kl"], random_kl=random["accumulated_kl"],
                    fixed_minus_reactive=fixed["accumulated_kl"] - reactive["accumulated_kl"],
                    fixed_minus_random=fixed["accumulated_kl"] - random["accumulated_kl"],
                    reactive_forced_high=reactive["forced_high_steps"],
                    reactive_threshold_selected_high=reactive["threshold_selected_high_steps"],
                    wall_s={policy: ledger["whole_policy_wall_s"] for policy, ledger in arms.items()}))
    terminal = json.loads((path / "COMPLETE.json").read_text()) if (path / "COMPLETE.json").exists() else {}
    status = "INVALID" if invalid else "PARTIAL" if missing or terminal.get("status") != "COMPLETE" else "COMPLETE"
    report = dict(status=status, missing=missing, invalid=invalid, per_document_repeat=rows,
                  pooled=None, document_means=None, new_budget_oracle="UNRUN",
                  strongest_periodic_phase_baseline="UNRUN; fixed phase zero is preregistered only",
                  positive_claim_ceiling="PROVISIONAL_SELECTOR_SIGNAL")
    if status == "COMPLETE":
        denominator = sum(row["fixed_kl"] for row in rows)
        means = [{"document": index, **{key: statistics.mean(row[key] for row in rows if row["document"] == index)
                 for key in ("fixed_kl", "reactive_kl", "random_kl", "fixed_minus_reactive", "fixed_minus_random")}}
                 for index in range(len(documents))]
        report["document_means"] = means
        report["pooled"] = {"fixed_kl_sum": denominator, "zero_denominator": denominator == 0}
        for key in ("fixed_minus_reactive", "fixed_minus_random"):
            values = [row[key] for row in means]
            normalized = [row[key] / row["fixed_kl"] for row in means if row["fixed_kl"] > 0]
            report["pooled"][key] = dict(ratio_of_sums=sum(row[key] for row in rows)/denominator if denominator else None,
                median_document_mean_delta=statistics.median(values),
                median_document_relative_gain=statistics.median(normalized) if len(normalized) == len(means) else None,
                zero_denominator_documents=sum(row["fixed_kl"] == 0 for row in means),
                positive=sum(v > 0 for v in values), negative=sum(v < 0 for v in values), zero=sum(v == 0 for v in values))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir), indent=2, allow_nan=False))
