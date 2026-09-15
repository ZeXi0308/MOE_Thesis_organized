#!/usr/bin/env python3
"""PLTB matched re-evaluation launcher notes + offline conflict table.

Writes a conflict summary from *existing* LLM-jp runs (INT4 vs MXFP4,
different datasets/offsets) and emits the exact matched-protocol commands
for GPU. Does not load models.
"""
from __future__ import annotations


# --- shared-lib bootstrap (auto) ---
import sys
from pathlib import Path as _Path

def _ensure_shared_on_path() -> None:
    here = _Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / "experiments" / "shared"
        if (cand / "capture_moe.py").exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        if (p / "capture_moe.py").exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return

_ensure_shared_on_path()
del _ensure_shared_on_path, _Path
# --- end bootstrap ---

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "pltb_matched_conflict_summary_2026-07-21"


def load_paired(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [
        {
            "label": "early_int4_wikitext2_off128",
            "path": ROOT / "outputs/paper_validation/llmjp_top16_layer_budget_n32/paired_bootstrap_vs_fixed.csv",
            "tail_precision": "int4",
            "dataset": "wikitext2",
            "test_offset": 128,
        },
        {
            "label": "jul20_mxfp4_wikitext2_docs_off20",
            "path": ROOT / "outputs/llmjp_layer_budget_mxfp4_cal16_n32_2026-07-20/paired_bootstrap_vs_fixed.csv",
            "tail_precision": "mxfp4",
            "dataset": "wikitext2_docs",
            "test_offset": 20,
        },
    ]
    rows = []
    for run in runs:
        df = load_paired(run["path"])
        for _, r in df.iterrows():
            delta = float(r["reference_minus_candidate_kl"])
            ci_low = float(r["ci_low"])
            ci_high = float(r["ci_high"])
            if ci_low > 0:
                verdict = "sig_better"
            elif ci_high < 0:
                verdict = "sig_worse"
            else:
                verdict = "insignificant"
            rows.append({
                **{k: run[k] for k in ("label", "tail_precision", "dataset", "test_offset")},
                "candidate": r["candidate"],
                "reference_minus_candidate_kl": delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "verdict": verdict,
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "existing_run_conflict_table.csv", index=False)

    n_better = int((summary["verdict"] == "sig_better").sum())
    n_worse = int((summary["verdict"] == "sig_worse").sum())
    universal_kill_valid = n_better == 0
    decision = {
        "universal_never_beats_fixed_VALID": universal_kill_valid,
        "n_sig_better_cells": n_better,
        "n_sig_worse_cells": n_worse,
        "status": (
            "WITHDRAW_UNIVERSAL_KILL"
            if not universal_kill_valid
            else "UNIVERSAL_KILL_CONSISTENT_WITH_EXISTING_TABLE"
        ),
        "matched_gpu_status": "PENDING_GPU",
        "matched_protocol": {
            "model": "llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M",
            "dataset": "wikitext2_docs",
            "calibration_samples": 16,
            "calibration_offset": 0,
            "test_samples": 32,
            "test_offset": 128,
            "base_tail": 8,
            "tail_precisions": ["int4", "mxfp4"],
        },
        "commands": [
            (
                "python run_layer_budget_experiment.py "
                "--model llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M "
                "--tail-precision int4 --dataset wikitext2_docs "
                "--calibration-samples 16 --calibration-offset 0 "
                "--test-samples 32 --test-offset 128 --base-tail 8 "
                "--output-dir outputs/llmjp_layer_budget_int4_matched_2026-07-21"
            ),
            (
                "python run_layer_budget_experiment.py "
                "--model llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M "
                "--tail-precision mxfp4 --dataset wikitext2_docs "
                "--calibration-samples 16 --calibration-offset 0 "
                "--test-samples 32 --test-offset 128 --base-tail 8 "
                "--output-dir outputs/llmjp_layer_budget_mxfp4_matched_2026-07-21"
            ),
        ],
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    lines = [
        "# PLTB Matched Conflict Summary (2026-07-21)",
        "",
        f"Existing LLM-jp cells: sig_better={n_better}, sig_worse={n_worse}.",
        f"**Universal 'never beats fixed' claim valid?** `{universal_kill_valid}`",
        "",
        "Early INT4 (`wikitext2` offset 128) and Jul-20 MXFP4 (`wikitext2_docs` offset 20) "
        "disagree in sign. Matched GPU reruns are required before any final label "
        "(`CONDITIONAL_FORMAT` / KILLED / open).",
        "",
        "## Existing conflict table",
        "",
        "| label | precision | candidate | delta | CI | verdict |",
        "|---|---|---|---:|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['label']} | {r['tail_precision']} | {r['candidate']} | "
            f"{r['reference_minus_candidate_kl']:.6f} | "
            f"[{r['ci_low']:.6f}, {r['ci_high']:.6f}] | {r['verdict']} |"
        )
    lines.append("")
    lines.append("## Matched GPU commands (PENDING)")
    lines.append("")
    for cmd in decision["commands"]:
        lines.append(f"```bash\n{cmd}\n```")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
