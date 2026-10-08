#!/usr/bin/env python3
"""Plot separate request-weighted H1 off/on ECDFs for the two audited blocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

cache = Path(tempfile.gettempdir()) / "moe_h1_pair_plot_cache"
cache.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache))
os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ARMS = ("off", "on")
STYLES = {"off": ("Recheck off", "#0072b2", "-"),
          "on": ("Recheck on", "#d55e00", "--")}
METRICS = (
    ("max_gap_s", "Maximum host-return generation gap", "s; log x"),
    ("actual_completion_flow_s", "Arrival-to-completion flow", "s"),
    ("ttft_s", "Time to first token", "s"),
)
EXPECTED_STATUS = "H1_PERFORMANCE_PAIR_COMPLETE"
EXPECTED_MANIFEST_SHA = "4231a687db066f113fcc14676f91e8e5825be05b76ef0834112e053726aa13aa"
EXPECTED_PROTOCOL_SHA = "3a36ae6e42c33eacfd0623c0f3448641ebe5adf75e42e99890c070931662f8f5"
EXPECTED_QUALIFICATION_SHA = "7a3b323673ac7b42d666f758de2109ad21ed625df9957bed8cfd149a75437f08"
EXPECTED_PLAN_SHA = {
    1: "2cbf257b30202ac8cc5e8196c988b741805fb69e882d5eb45e69e1c3d0b0714f",
    2: "51abe14ad552fb9efa2ebd305b6725e07d1999a8d3cc1b4e966db68dfd16a366",
}
OUTPUT_DIR_NAME = "A_H1_PERFORMANCE_PAIRED_ECDF_R01_20261001"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_audit(path: Path, expected_sha: str, block: int) -> dict:
    if not path.is_file() or path.is_symlink() or sha256(path) != expected_sha:
        raise ValueError(f"block {block} audit missing, symlinked or SHA-256 drifted")
    audit = json.loads(path.read_text())
    if (audit.get("schema_version") != 1 or audit.get("status") != EXPECTED_STATUS or
            audit.get("block_index") != block or
            audit.get("plan_sha256") != EXPECTED_PLAN_SHA[block] or
            audit.get("package_manifest_sha256") != EXPECTED_MANIFEST_SHA or
            audit.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA or
            audit.get("qualification_audit_sha256") != EXPECTED_QUALIFICATION_SHA or
            set(audit.get("metrics", {})) != set(ARMS)):
        raise ValueError(f"block {block} is not the complete frozen H1 pair")
    return audit


def validate_cohort(audits: dict[int, dict]) -> None:
    reference = None
    for block in (1, 2):
        audit = audits[block]
        for arm in ARMS:
            metrics = audit["metrics"][arm]
            if ((metrics.get("expected_requests"), metrics.get("completed"),
                 metrics.get("failed"), metrics.get("unfinished")) != (128, 128, 0, 0)):
                raise ValueError(f"block {block}/{arm}: incomplete request cohort")
            requests = metrics.get("requests")
            if not isinstance(requests, list) or len(requests) != 128:
                raise ValueError(f"block {block}/{arm}: request rows missing")
            identity = []
            for row in requests:
                if row.get("status") != "completed" or row.get("completed") is not True:
                    raise ValueError(f"block {block}/{arm}: incomplete request row")
                identity.append((row["request_id"], row["document_id"], row["prompt_sha256"],
                                 row["arrival_s"], row["max_output"]))
                for key in ("max_gap_s", "actual_completion_flow_s", "ttft_s"):
                    value = row.get(key)
                    if value is not None and (type(value) not in (int, float) or
                                              not math.isfinite(value) or value < 0):
                        raise ValueError(f"block {block}/{arm}: invalid {key}")
            identity.sort()
            if len({row[0] for row in identity}) != 128 or (reference is not None and identity != reference):
                raise ValueError(f"block {block}/{arm}: request identity differs")
            reference = identity


def ecdf(requests: list[dict], field: str) -> tuple[list[float], list[float], int]:
    """Undefined observations retain their 1/128 denominator weight off the x-axis."""
    values = sorted(row[field] for row in requests if row[field] is not None)
    if field == "max_gap_s" and any(value <= 0 for value in values):
        raise ValueError("log-scale generation gap must be positive when observed")
    return values, [(index + 1) / 128 for index in range(len(values))], 128 - len(values)


def make_figure(audits: dict[int, dict], source_short: str):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "svg.hashsalt": source_short})
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.25), sharey=True)
    global_max = {}
    for field, _, _ in METRICS:
        values = [row[field] for block in (1, 2) for arm in ARMS
                  for row in audits[block]["metrics"][arm]["requests"]
                  if row[field] is not None]
        if not values:
            raise ValueError(f"all four arms have undefined {field}")
        global_max[field] = max(values)
        if field == "max_gap_s":
            global_max["gap_min"] = min(values)

    for row_index, block in enumerate((1, 2)):
        for column, (field, title, unit) in enumerate(METRICS):
            ax = axes[row_index, column]
            undefined = {}
            for arm in ARMS:
                label, color, style = STYLES[arm]
                values, fractions, missing = ecdf(audits[block]["metrics"][arm]["requests"], field)
                undefined[arm] = missing
                if values:
                    x0 = max(global_max["gap_min"] / 1.4, 1e-6) if field == "max_gap_s" else 0
                    ax.step([x0, *values], [0, *fractions], where="post",
                            color=color, linestyle=style, linewidth=2.0, label=label)
            if field == "max_gap_s":
                ax.set_xscale("log")
                ax.set_xlim(max(global_max["gap_min"] / 1.4, 1e-6),
                            global_max[field] * 1.15)
            else:
                ax.set_xlim(0, global_max[field] * 1.05)
            if field == "max_gap_s" or any(undefined.values()):
                ax.text(.98, .055,
                        f"Undefined: off {undefined['off']}, on {undefined['on']} / 128",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                        color="#485665")
            ax.set_ylim(0, 1.015)
            ax.grid(alpha=.17, linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                ax.set_title(title, fontsize=11, fontweight="semibold")
            if row_index == 1:
                ax.set_xlabel(unit)
            if column == 0:
                order = "off → on" if block == 1 else "on → off"
                ax.set_ylabel(f"Block {block} ({order})\nFraction of requests ≤ x")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("H1 direct recheck · separate off/on request distributions",
                 y=.975, fontsize=14, fontweight="semibold")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .93),
               ncol=2, frameon=False)
    fig.text(.5, .095,
             "Each request weighs 1/128 in its own arm/block; undefined values stay in the denominator. Host-return generation gap excludes TTFT.",
             ha="center", fontsize=9, color="#3d4b57")
    fig.text(.5, .065,
             "Marginal ECDFs do not show paired request deltas. Seen H128, natural outputs; two single ordered pairs, no pooling, equal-work, CI or blind claim.",
             ha="center", fontsize=9, color="#3d4b57")
    fig.text(.5, .035, f"Pinned audit SHA prefixes: {source_short}",
             ha="center", fontsize=8, color="#687782")
    fig.subplots_adjust(left=.075, right=.985, top=.84, bottom=.17,
                        hspace=.33, wspace=.18)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--b1-sha256", required=True)
    parser.add_argument("--b2", type=Path, required=True)
    parser.add_argument("--b2-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR_NAME))
    args = parser.parse_args()
    if args.b1.resolve() == args.b2.resolve():
        raise ValueError("block 1 and block 2 must use distinct audited files")
    audits = {1: read_audit(args.b1, args.b1_sha256, 1),
              2: read_audit(args.b2, args.b2_sha256, 2)}
    validate_cohort(audits)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    short = f"B1 {args.b1_sha256[:12]} · B2 {args.b2_sha256[:12]}"
    fig = make_figure(audits, short)
    png = args.output_dir / "h1_pair_request_ecdf.png"
    svg = args.output_dir / "h1_pair_request_ecdf.svg"
    fig.savefig(png, dpi=220, facecolor="white", metadata={"Software": "matplotlib"})
    fig.savefig(svg, facecolor="white", metadata={"Date": None})
    plt.close(fig)
    manifest = {
        "schema_version": 1,
        "audit_sources": [{"block_index": 1, "path": str(args.b1), "sha256": args.b1_sha256},
                          {"block_index": 2, "path": str(args.b2), "sha256": args.b2_sha256}],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "qualification_audit_sha256": EXPECTED_QUALIFICATION_SHA,
        "plot_script": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "outputs": [{"path": png.name, "sha256": sha256(png)},
                    {"path": svg.name, "sha256": sha256(svg)}],
        "weighting": "Each completed request weighs 1/128 in its own arm and block; undefined values remain in the denominator.",
        "limits": "Separate single ordered pairs on seen H128; natural outputs may differ; no pooling, CI or blind-test claim.",
    }
    hashes = args.output_dir / "source_hashes.json"
    hashes.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "source_hashes": str(hashes)}))


if __name__ == "__main__":
    main()
