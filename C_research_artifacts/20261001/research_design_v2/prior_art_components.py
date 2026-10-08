#!/usr/bin/env python3
"""CPU-only action witnesses, not a serving controller or CacheOPT reproduction.

Source: CacheOPT v1 Sec. 3.3.1, https://arxiv.org/html/2503.13773v1#S3.SS3.SSS1
Implements only full-demand pairwise embedding with predicted output replaced by
an online hard output cap, followed by the unallocated-capacity path. Omits
CacheOPT's prediction, SLO ordering, cuts, proactive allocation and preemption.
M0 below is solely a reference for two separating, hand-constructed states.
No GPU, network, trace replay, measured latency or native allocator is used.
"""

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path


def blocks(tokens: int, block_size: int) -> int:
    return (tokens + block_size - 1) // block_size


@dataclass(frozen=True)
class Decoder:
    prompt: int
    output: int
    hard_cap: int

    def __post_init__(self):
        if self.prompt <= 0 or not 0 < self.output < self.hard_cap:
            raise ValueError("fixture must be an unfinished pure decoder")

    @property
    def conservative_used(self):
        # M0's P+O convention; native computed KV can instead be P+O-1.
        return self.prompt + self.output

    @property
    def remaining(self):
        return self.hard_cap - self.output


@dataclass(frozen=True)
class Candidate:
    prompt: int
    hard_cap: int

    def __post_init__(self):
        if self.prompt <= 0 or self.hard_cap <= 0:
            raise ValueError("positive prompt and output cap required")


def cacheopt_pairwise_hard_cap(
    residents, candidate, capacity_blocks, block_size=16,
    buffer_tokens=0, native_legal=True, used_offset=0,
):
    """One-candidate, full-demand component. Reservations are full hard caps.

    Require a host to retain its allocation through candidate completion. The
    paper condition is a_j-(u_j+s_i^o)-(s_i^p+s_i^o)>=b. Round candidate demand
    and host reservations to blocks. used_offset=-1 checks native C=P+O-1.
    native_legal is a common supplied predicate, NOT a native legality emulator.
    Multiple nested placements / logical interval ownership are unsupported.
    """
    if buffer_tokens < 0 or block_size <= 0 or used_offset not in (0, -1):
        raise ValueError("invalid component parameter")
    candidate_demand = blocks(candidate.prompt + candidate.hard_cap, block_size)
    reservations = [blocks(d.prompt + d.hard_cap, block_size) for d in residents]
    unallocated = capacity_blocks - sum(reservations)
    if unallocated < 0:
        raise ValueError("component requires feasible original reservations")
    checks = []
    for i, (d, reservation) in enumerate(zip(residents, reservations)):
        used = d.conservative_used + used_offset
        margin = (reservation * block_size - used - candidate.hard_cap
                  - candidate_demand * block_size - buffer_tokens)
        checks.append({"host": i, "margin_tokens": margin,
                       "host_remaining_iterations": d.remaining,
                       "candidate_cap": candidate.hard_cap,
                       "feasible": margin >= 0 and d.remaining >= candidate.hard_cap})
    hosts = [c for c in checks if c["feasible"]]
    # The source chooses minimum remaining allocated KV among feasible hosts.
    hosts.sort(key=lambda c: (
        reservations[c["host"]] * block_size
        - residents[c["host"]].conservative_used - used_offset,
        c["host"],
    ))
    if not native_legal:
        mode = "reject_native"
    elif hosts:
        mode = "embed_full_demand"
    elif unallocated >= candidate_demand:
        mode = "unallocated_full_demand"
    else:
        mode = "reject_full_demand_component"
    return {"accept": mode in ("embed_full_demand", "unallocated_full_demand"),
            "mode": mode, "host": hosts[0]["host"] if hosts else None,
            "candidate_blocks": candidate_demand,
            "unallocated_reservation_blocks": unallocated,
            "host_checks": checks}


def m0_fixture_action(residents, candidate, capacity_blocks, block_size=16,
                      native_legal=True):
    endpoints = sorted({0, *(d.remaining for d in residents)})
    occupancy = lambda t: sum(
        blocks(d.conservative_used + t, block_size)
        for d in residents if t <= d.remaining
    )
    curve = {str(t): occupancy(t) for t in endpoints}
    peak = max(curve.values(), default=0)
    # Single-fixture verification that the reference endpoint value is correct.
    assert peak == max(occupancy(t) for t in range(max(endpoints) + 1))
    candidate_blocks = blocks(candidate.prompt + candidate.hard_cap, block_size)
    return {"accept": native_legal and peak + candidate_blocks <= capacity_blocks,
            "resident_peak_blocks": peak, "candidate_blocks": candidate_blocks,
            "total_certificate_blocks": peak + candidate_blocks,
            "endpoint_occupancy_blocks": curve}


def run_witnesses():
    cases = [
        ("pooled_retirement_accepts_pairwise_rejects",
         [Decoder(16, 16, 144), Decoder(16, 128, 144)], Candidate(96, 32), 21,
         True, False),
        ("pairwise_candidate_retirement_accepts_m0_rejects",
         [Decoder(16, 16, 144)], Candidate(16, 16), 10,
         False, True),
    ]
    result = {"evidence_tier": "STRUCTURAL_CPU_COMPONENT",
              "source": "https://arxiv.org/html/2503.13773v1#S3.SS3.SSS1",
              "scope": "hard-cap full-demand embedding component; not full CacheOPT",
              "native_legal": "fixture predicate true; native execution UNRUN",
              "block_size": 16, "witnesses": []}
    for name, residents, candidate, capacity, expected_m0, expected_pair in cases:
        m0 = m0_fixture_action(residents, candidate, capacity)
        variants = []
        for buffer in (0, 8, 16):
            for offset in (0, -1):
                action = cacheopt_pairwise_hard_cap(
                    residents, candidate, capacity, buffer_tokens=buffer,
                    used_offset=offset,
                )
                assert action["accept"] is expected_pair
                variants.append({"buffer_tokens": buffer, "used_offset": offset,
                                 "action": action})
        assert m0["accept"] is expected_m0
        assert m0["accept"] != variants[0]["action"]["accept"]
        # Both components must reject if the shared native predicate fails.
        assert not m0_fixture_action(residents, candidate, capacity,
                                     native_legal=False)["accept"]
        assert not cacheopt_pairwise_hard_cap(residents, candidate, capacity,
                                              native_legal=False)["accept"]
        result["witnesses"].append({"name": name,
            "residents": [asdict(d) for d in residents],
            "candidate": asdict(candidate), "capacity_blocks": capacity,
            "m0": m0, "pairwise_variants": variants})
    result["assertions_passed"] = True
    result["conclusion"] = (
        "M0 and this pairwise full-demand component are not action-equivalent "
        "and neither acceptance set contains the other. This says nothing "
        "about complete CacheOPT or full-request performance.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        help="write a new result path; existing files are refused")
    args = parser.parse_args()
    result = run_witnesses()
    if args.out:
        with args.out.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
    print(json.dumps({"witnesses": len(result["witnesses"]),
                      "assertions_passed": result["assertions_passed"],
                      "result_path": str(args.out) if args.out else None},
                     ensure_ascii=False))
