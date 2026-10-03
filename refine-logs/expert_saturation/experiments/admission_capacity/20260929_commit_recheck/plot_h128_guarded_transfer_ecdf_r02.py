#!/usr/bin/env python3
"""Plot request-weighted H128 ECDFs from the pinned guarded-transfer audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

cache_root = Path(tempfile.gettempdir()) / "moe_h128_plot_cache"
cache_root.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


AUDIT_NAME = "A_H128_GUARDED_TRANSFER_AUDIT_R02_20260930.json"
AUDIT_SHA256 = "e5c5ae271e14159e6aceb1b2b2b0bd7324c9bcc92c2ccdc830488c2838415376"
SELECTION_NAME = "A_G64_GUARDED_FOUR_POINT_SELECTION_20260930.json"
OUTPUT_DIR_NAME = "A_H128_GUARDED_TRANSFER_ECDF_R02_20260930"
ARMS = ("native_full_native", "eager", "ltr_t200_q1")
STYLES = {
    "native_full_native": ("Native full", "#596d7a", ":"),
    "eager": ("Eager selected save", "#0072b2", "-"),
    "ltr_t200_q1": ("LTR T200/Q1", "#d55e00", "--"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pinned(audit_path: Path, selection_path: Path) -> dict:
    if audit_path.name != AUDIT_NAME or sha256(audit_path) != AUDIT_SHA256:
        raise ValueError("H128 guarded transfer audit identity or SHA-256 differs")
    audit = json.loads(audit_path.read_text())
    if (audit.get("schema_version") != 1 or
            audit.get("status") != "H128_FROZEN_LTR_TRANSFER_COMPLETE" or
            audit.get("selected_arm") != "ltr_t200_q1" or
            set(audit.get("metrics", {})) != set(ARMS)):
        raise ValueError("H128 transfer is incomplete or has different arms")
    if (selection_path.name != SELECTION_NAME or
            sha256(selection_path) != audit.get("selection_sha256")):
        raise ValueError("frozen G64 selection provenance differs")

    cohort = None
    for arm in ARMS:
        metric = audit["metrics"][arm]
        if ((metric["expected_requests"], metric["completed"], metric["failed"],
             metric["unfinished"]) != (128, 128, 0, 0)):
            raise ValueError(f"{arm}: incomplete 128-request cohort")
        requests = metric["requests"]
        if len(requests) != 128 or not all(r["completed"] and r["status"] == "completed"
                                                 for r in requests):
            raise ValueError(f"{arm}: request completion differs")
        identity = sorted((r["request_id"], r["document_id"], r["prompt_sha256"],
                           r["arrival_s"], r["max_output"]) for r in requests)
        if len(set(row[0] for row in identity)) != 128 or (cohort is not None and identity != cohort):
            raise ValueError(f"{arm}: cohort identity differs")
        cohort = identity
        for field in ("max_gap_s", "actual_completion_flow_s"):
            if any(not isinstance(r[field], (int, float)) or
                   not math.isfinite(r[field]) or r[field] <= 0 for r in requests):
                raise ValueError(f"{arm}: {field} is missing or invalid")
    return audit


def figure(audit: dict):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.hashsalt": AUDIT_SHA256})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.35), sharey=True)
    panels = (
        ("max_gap_s", "Maximum generation gap per request (s; log scale)"),
        ("actual_completion_flow_s", "Arrival-to-completion flow per request (s)"),
    )
    for ax, (field, xlabel) in zip(axes, panels):
        for arm in ARMS:
            name, color, style = STYLES[arm]
            values = sorted(r[field] for r in audit["metrics"][arm]["requests"])
            x0 = .006 if field == "max_gap_s" else 0
            x = [x0, *values]
            y = [0, *(i / 128 for i in range(1, 129))]
            ax.step(x, y, where="post", color=color, linestyle=style,
                    linewidth=2.15, label=name)
        if field == "max_gap_s":
            ax.set_xscale("log")
            ax.set_xlim(.006, 16)
        else:
            ax.set_xlim(0, 75)
        ax.set_ylim(0, 1.015)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=.18, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fraction of requests ≤ x")
    fig.suptitle("H128 fixed LTR transfer · empirical request distributions",
                 fontsize=14, fontweight="semibold", y=.965)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .89),
               frameon=False, ncol=3)
    fig.text(.5, .095,
             "Each of 128 requests has equal weight in each arm. LTR vs eager: max gap lower for 92 requests, higher for 36; flow shorter for 48, longer for 80.",
             ha="center", fontsize=9, color="#3d4c58")
    fig.text(.5, .057,
             "Single ordered transfer on previously seen H128 inputs; natural output sequences may differ. No confidence intervals or blind-test claim.",
             ha="center", fontsize=9, color="#3d4c58")
    fig.text(.5, .025, f"Audit SHA-256: {AUDIT_SHA256}",
             ha="center", fontsize=7.5, color="#697783")
    fig.subplots_adjust(left=.075, right=.985, top=.77, bottom=.24, wspace=.12)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path(AUDIT_NAME))
    parser.add_argument("--selection", type=Path, default=Path(SELECTION_NAME))
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR_NAME))
    args = parser.parse_args()
    audit = read_pinned(args.audit, args.selection)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fig = figure(audit)
    png = args.output_dir / "h128_request_ecdf.png"
    svg = args.output_dir / "h128_request_ecdf.svg"
    fig.savefig(png, dpi=240, facecolor="white", metadata={"Software": "matplotlib"})
    fig.savefig(svg, facecolor="white", metadata={"Date": None})
    plt.close(fig)
    hashes = {
        "schema_version": 1,
        "audit_source": {"path": args.audit.as_posix(), "sha256": AUDIT_SHA256},
        "selection_source": {"path": args.selection.as_posix(),
                             "sha256": audit["selection_sha256"]},
        "plot_script": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "outputs": [{"path": png.name, "sha256": sha256(png)},
                    {"path": svg.name, "sha256": sha256(svg)}],
        "weighting": "Each completed request contributes 1/128 to each arm's empirical CDF.",
        "limits": "Single ordered transfer on seen inputs; natural outputs may differ; no CI or blind-test claim.",
    }
    (args.output_dir / "source_hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "arms": list(ARMS),
                      "source_hashes": str(args.output_dir / "source_hashes.json")}))


if __name__ == "__main__":
    main()
