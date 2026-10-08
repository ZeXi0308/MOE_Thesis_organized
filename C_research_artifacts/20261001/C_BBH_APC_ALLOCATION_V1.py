#!/usr/bin/env python3
"""Analyze native APC physical-block census and plot each arm's own host time."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ARMS = ("apc_off", "apc_on")
USABLE = 4096
FILES = ("status.json", "measured-steps.json", "prefix-cache-reset.json",
         "engine_args.json", "native-drain.json")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def snapshot(row: dict, arm: str) -> None:
    unique, shared, refs, free = (row[key] for key in
                                  ("unique_active_blocks", "shared_active_blocks",
                                   "active_block_references", "free_blocks"))
    require(all(type(value) is int for value in (unique, shared, refs, free))
            and 0 <= shared <= unique <= USABLE and 0 <= free <= USABLE
            and refs >= unique + shared and unique + free == USABLE
            and row["free_active_accounting_consistent"] is True,
            f"{arm}: physical free/refcount census inconsistent")


def analyze_arm(root: Path, arm: str) -> tuple[dict, list[dict]]:
    folder = root / arm
    docs = {name: json.loads((folder / name).read_text()) for name in FILES}
    status, trace, reset, args, drain = (docs[name] for name in FILES)
    require(status["status"] == "COMPLETE" and status["arm"] == arm
            and status["request_count"] == 32 and drain["status"] == "QUALIFIED"
            and drain["total_blocks"] == 4097 and drain["free_blocks"] == USABLE,
            f"{arm}: incomplete measurement or native KV drain")
    require(args["enable_prefix_caching"] is (arm == "apc_on")
            and args["max_num_seqs"] == 32 and args["max_num_batched_tokens"] == 1024
            and args["max_model_len"] == 4096,
            f"{arm}: engine/APC resource arguments changed")
    require(reset["reset_succeeded"] is True
            and reset["drained_before"]["status"] == "QUALIFIED"
            and reset["drained_after"]["status"] == "QUALIFIED",
            f"{arm}: warmup prefix reset or drain failed")
    for name in ("before", "after"):
        snapshot(reset[name], arm)
        require(reset[name]["unique_active_blocks"] == 0,
                f"{arm}: request-owned KV remained before/after reset")
    after = reset["after"]
    require(after["cached_hash_keys"] == after["hashed_blocks"] ==
            after["auxiliary_hash_blocks"] == 0,
            f"{arm}: prefix hash state not cleared before measurement")
    states, steps = trace["schedule_states"], trace["steps"]
    require(isinstance(states, list) and states and isinstance(steps, list) and steps,
            f"{arm}: empty measured native snapshots")
    require(all(row["schedule_index"] == index for index, row in enumerate(states)),
            f"{arm}: schedule snapshot indices changed")
    require(all(states[i]["return_s"] <= states[i + 1]["return_s"]
                for i in range(len(states) - 1)),
            f"{arm}: schedule host time regressed")
    for row in states + steps:
        snapshot(row, arm)
    lookup = trace["lookup_events"]
    hit_events = [row for row in lookup if row["hit_tokens"] > 0]
    require(all(type(row["hit_tokens"]) is int and row["hit_tokens"] >= 0
                for row in lookup), f"{arm}: invalid lookup hit count")
    hit_tokens = sum(row["hit_tokens"] for row in lookup)
    hit_requests = len({row["request_id"] for row in hit_events})
    require(status["lookup_events"] == len(lookup)
            and status["total_lookup_hit_tokens"] == hit_tokens
            and status["requests_with_lookup_hits"] == hit_requests
            and status["peak_unique_active_physical_blocks"] ==
                max(row["unique_active_blocks"] for row in states)
            and status["peak_shared_active_blocks"] ==
                max(row["shared_active_blocks"] for row in states),
            f"{arm}: cell status differs from measured census")
    witness = trace["shared_witness"]
    require(witness == status["shared_witness"], f"{arm}: shared witness receipt differs")
    if witness is not None:
        require(isinstance(witness["request_ids"], list)
                and len(set(witness["request_ids"])) >= 2
                and witness["native_ref_cnt"] >= 2
                and any(row["schedule_index"] == witness["schedule_index"]
                        and row["shared_active_blocks"] > 0
                        for row in states),
                f"{arm}: shared witness lacks two request owners")
    first_shared = next((row["schedule_index"] for row in states
                         if row["shared_active_blocks"] > 0), None)
    summary = dict(arm=arm,
        sources_sha256={name: sha(folder / name) for name in FILES},
        measured_schedule_snapshots=len(states), measured_engine_steps=len(steps),
        all_free_active_accounting_consistent=True,
        peak_unique_active_physical_blocks=max(row["unique_active_blocks"] for row in states),
        peak_shared_active_blocks=max(row["shared_active_blocks"] for row in states),
        peak_native_active_block_references=max(row["active_block_references"] for row in states),
        peak_excess_native_references=max(
            row["active_block_references"] - row["unique_active_blocks"] for row in states),
        minimum_free_blocks=min(row["free_blocks"] for row in states),
        schedule_snapshots_with_shared_refs=sum(row["shared_active_blocks"] > 0 for row in states),
        first_shared_ref_snapshot_index=first_shared,
        first_shared_ownership_witness=witness,
        verified_at_least_two_request_owners=witness is not None,
        lookup_calls=len(lookup), lookup_hit_calls=len(hit_events),
        lookup_hit_tokens=hit_tokens, requests_with_lookup_hits=hit_requests,
        allocation_calls=len(trace["allocation_events"]),
        successful_allocation_calls=sum(row["succeeded"] for row in trace["allocation_events"]),
        preemptions=status["preemptions"], output_tokens=status["output_tokens"],
        measurement_observation_end_s=status["observation_end_s"],
        reset_succeeded=True, reset_after_empty_and_hash_free=True,
        final_native_drain_qualified=True)
    return summary, states


def plot(states_by_arm: dict[str, list[dict]], summaries: dict, png: Path, svg: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.7), sharey=True)
    fig.subplots_adjust(left=.11, right=.88, bottom=.16, top=.83, hspace=.34)
    for ax, arm, color in zip(axes, ARMS, ("#315F91", "#188679")):
        states = states_by_arm[arm]
        x = [row["return_s"] for row in states]
        occupied = [row["unique_active_blocks"] for row in states]
        shared = [row["shared_active_blocks"] for row in states]
        ax.plot(x, occupied, color=color, linewidth=1.5, label="Unique active native blocks")
        ax.axhline(USABLE, color="#A9444E", linestyle=(0, (5, 3)),
                   linewidth=1, label="4,096 usable blocks")
        ax.set_ylim(0, 4290)
        ax.set_xlim(left=0, right=max(x) * 1.02)
        ax.set_ylabel("Physical KV blocks")
        ax.grid(axis="y", color="#E5EAEE", linewidth=.8)
        ax.set_axisbelow(True)
        ax.set_title(f"{'APC off' if arm == 'apc_off' else 'APC on'}  ·  "
                     f"peak unique {summaries[arm]['peak_unique_active_physical_blocks']:,}  ·  "
                     f"min free {summaries[arm]['minimum_free_blocks']:,}",
                     loc="left", fontsize=11, color="#203447")
        ax.set_xlabel("Host time within this arm (s)")
        if arm == "apc_on":
            twin = ax.twinx()
            twin.plot(x, shared, color="#C98E19", alpha=.85, linewidth=1.4)
            twin.set_ylabel("Blocks with refcount ≥ 2", color="#9A6A12")
            twin.tick_params(axis="y", colors="#9A6A12")
            twin.set_ylim(0, max(1, max(shared) * 1.35))
            witness = summaries[arm]["first_shared_ownership_witness"]
            if witness is not None:
                sample = states[witness["schedule_index"]]
                ax.scatter([sample["return_s"]], [sample["unique_active_blocks"]],
                           s=55, marker="o", facecolor="#D79B2B", edgecolor="white",
                           linewidth=.8, zorder=5, label="Two-owner native witness")
    axes[0].legend(handles=[
        Line2D([0], [0], color="#315F91", lw=1.6, label="Unique active blocks"),
        Line2D([0], [0], color="#A9444E", lw=1, linestyle="--",
               label="4,096-block capacity")],
        loc="lower left", frameon=False, fontsize=9)
    axes[1].legend(handles=[
        Line2D([0], [0], color="#188679", lw=1.6, label="Unique active blocks"),
        Line2D([0], [0], color="#C98E19", lw=1.4,
               label="Shared-ref blocks (right axis)"),
        Line2D([0], [0], color="#A9444E", lw=1, linestyle="--",
               label="4,096-block capacity"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D79B2B",
               markeredgecolor="white", markersize=7,
               label="Two-owner native witness")],
        loc="lower left", frameon=False, fontsize=9)
    fig.text(.11, .945, "BBH APC physical KV allocation qualification",
             fontsize=16, fontweight="bold", color="#172C3D")
    fig.text(.11, .902,
             "32 source-first geometric-shapes requests per arm  ·  native APC off / on  ·  "
             "each trace has its own time origin",
             fontsize=9.5, color="#536576")
    fig.text(.11, .055,
             "After-schedule native refcount census excludes null and free hashed blocks. "
             "Lookup hits alone do not establish sharing; the marked witness names ≥2 request owners. "
             "Instrumented host time is descriptive, not a performance comparison or a GPU-tensor-byte claim.",
             fontsize=8.6, color="#607384", wrap=True)
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()
    require(all(not path.exists() for path in
                (args.output_json, args.png, args.svg)), "analysis outputs must be new")
    summaries, states = {}, {}
    for arm in ARMS:
        summaries[arm], states[arm] = analyze_arm(args.root, arm)
    report = dict(schema="c-bbh-apc-allocation-v1", source_root=str(args.root),
        arms=summaries,
        scope="Native physical-block/refcount allocation qualification only. "
              "Lookup events are diagnostic, not proof of sharing; ownership witness "
              "requires two request IDs. Host times are independently originated and "
              "descriptive; no timing performance or raw GPU tensor bytes inferred.")
    with args.output_json.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    plot(states, summaries, args.png, args.svg)
    print(json.dumps({"output_json": str(args.output_json),
                      "png": str(args.png), "svg": str(args.svg),
                      "arms": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()
