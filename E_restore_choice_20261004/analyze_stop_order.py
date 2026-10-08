#!/usr/bin/env python3
"""Read two completed raw traces; compare stop order and a peer's replay.

All step indices are zero-based and token intervals are half-open. This is
descriptive trace analysis, with no latency correction or significance test.
"""
import argparse
import bisect
import hashlib
import json
from pathlib import Path


class Trace:
    def __init__(self, path, peer):
        self.path = Path(path)
        data = self.path.read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.raw = json.loads(data)
        self.ends = [s["end_s"] for s in self.raw["steps"]]
        assert len(self.ends) == len(self.raw["scheduler_steps"])
        self.requests = {r["external_id"]: r for r in self.raw["requests"]}
        self.ids = {r["request_id"]: r["external_id"] for r in self.raw["requests"]}
        assert len(self.requests) == len(self.raw["requests"])
        assert all(r["completed"] for r in self.requests.values())
        self.finish = {k: self.step(r["completion_s"]) for k, r in self.requests.items()}
        self.token_steps = {}
        for k, r in self.requests.items():
            assert len(r["token_times_s"]) == len(r["output_token_ids"])
            self.token_steps[k] = [self.step(t) for t in r["token_times_s"]]
            assert all(self.ends[s] == t for s, t in zip(self.token_steps[k], r["token_times_s"]))
            assert self.ends[self.finish[k]] == r["completion_s"]
        self.schedule = [
            [(self.ids[x["request_id"]], x["start_computed"], x["end_computed"],
              x["count"], x["known_tokens"], x["generated_tokens"])
             for x in s["scheduled"]]
            for s in self.raw["scheduler_steps"]
        ]
        self.request_schedule = {key: [] for key in self.requests}
        for step, schedule in enumerate(self.schedule):
            for key, start, end, count, known, generated in schedule:
                self.request_schedule[key].append((step, start, end, count, known, generated))
        self.request_work = {}
        for key, rows in self.request_schedule.items():
            seen, scheduled, repeated = set(), 0, 0
            for _, start, end, count, _, _ in rows:
                assert end - start == count
                scheduled += count
                repeated += sum(p in seen for p in range(start, end))
                seen.update(range(start, end))
            self.request_work[key] = dict(scheduled=scheduled, unique_positions=len(seen), repeated_positions=repeated)
        self.peer_rows = []
        seen = set()
        for step, schedule in enumerate(self.schedule):
            for key, start, end, count, known, generated in schedule:
                if key != peer:
                    continue
                assert end - start == count
                duplicate = sum(p in seen for p in range(start, end))
                seen.update(range(start, end))
                self.peer_rows.append(dict(step=step, start=start, end=end, count=count,
                                           known=known, generated=generated,
                                           previously_scheduled_positions=duplicate))
        self.peer_work = dict(scheduled=sum(x["count"] for x in self.peer_rows),
                              unique_positions=len(seen),
                              repeated_positions=sum(x["previously_scheduled_positions"] for x in self.peer_rows))

    def step(self, time):
        step = bisect.bisect_left(self.ends, time)
        assert step < len(self.ends), time
        assert self.raw["steps"][step]["start_s"] <= time <= self.ends[step], (step, time)
        return step

    def count(self, external_id, step):
        return bisect.bisect_right(self.token_steps[external_id], step)

    def snapshot(self, step):
        return dict(step=step,
                    finished_count=sum(s <= step for s in self.finish.values()),
                    emitted_tokens=sum(self.count(k, step) for k in self.requests))


def first_content_difference(a, b):
    rows = []
    for key in a.requests:
        x, y = a.requests[key], b.requests[key]
        for i, (u, v) in enumerate(zip(x["output_token_ids"], y["output_token_ids"])):
            if u != v:
                sa, sb = a.token_steps[key][i], b.token_steps[key][i]
                rows.append(dict(external_id=key, output_index=i, host_step=sa,
                                 other_step=sb, host_token=u, other_token=v,
                                 both_observed_by_step=max(sa, sb)))
                break
    return min(rows, key=lambda x: (x["both_observed_by_step"], x["external_id"])) if rows else None


def request_comparison(a, b, key, step):
    x, y = a.requests[key], b.requests[key]
    na, nb = a.count(key, step), b.count(key, step)
    equal_sequence = x["output_token_ids"] == y["output_token_ids"]
    return dict(external_id=key, host_finish_step=a.finish[key], other_finish_step=b.finish[key],
                host_length=len(x["output_token_ids"]), other_length=len(y["output_token_ids"]),
                host_finish_reason=x["finish_reason"], other_finish_reason=y["finish_reason"],
                full_token_sequence_equal=equal_sequence,
                host_emitted_at_step=na, other_emitted_at_step=nb,
                host_latest_token=x["output_token_ids"][na - 1] if na else None,
                other_latest_token=y["output_token_ids"][nb - 1] if nb else None,
                classification=("same complete sequence; completion progress differs" if equal_sequence
                                else "different terminal lengths" if len(x["output_token_ids"]) != len(y["output_token_ids"])
                                else "same final length, different contents; do not infer length change"))


def peer_events(trace, peer, fork):
    preemptions = []
    for p in trace.raw["preemptions"]:
        if trace.ids[p["request_id"]] != peer or trace.step(p["time_s"]) < fork:
            continue
        step = trace.step(p["time_s"])
        replay = []
        for row in trace.peer_rows:
            if row["step"] < step:
                continue
            replay.append(row)
            if row["end"] >= p["known_tokens"]:
                break
        preemptions.append(dict(step=step, known_tokens=p["known_tokens"],
                                generated_tokens=p["generated_tokens"], computed_tokens=p["computed_tokens"],
                                subsequent_recovery_rows=replay))
    commits = [dict(step=trace.step(c["allocation_s"]), event=c["event"],
                    known_tokens=c["known_tokens"], host_hit_tokens=c["host_hit_tokens"],
                    actual_action=c["actual_action"], fallback=c["fallback"])
               for c in trace.raw["commits"]
               if trace.ids[c["request_id"]] == peer and trace.step(c["allocation_s"]) >= fork]
    request = trace.requests[peer]
    all_commits = [c for c in trace.raw["commits"] if trace.ids[c["request_id"]] == peer]
    next_indices = [bisect.bisect_right(request["token_times_s"], c["decision_s"]) for c in all_commits]
    recovery = []
    for c, index in list(zip(all_commits, next_indices))[:3]:
        decision_step = trace.step(c["decision_s"])
        available = index < len(request["token_times_s"])
        output_step = trace.token_steps[peer][index] if available else None
        recovery.append(dict(event=c["event"], decision_step=decision_step,
            allocation_step=trace.step(c["allocation_s"]), known_tokens=c["known_tokens"],
            generated_tokens=c["generated_tokens"], host_hit_tokens=c["host_hit_tokens"],
            actual_action=c["actual_action"], external_tokens=c.get("external_tokens"),
            target_committed=c.get("target_committed"), next_output_index=index if available else None,
            next_output_step=output_step, censored=not available,
            decision_to_next_output_s=request["token_times_s"][index]-c["decision_s"] if available else None,
            commits_sharing_next_output=next_indices.count(index) if available else None,
            scheduled_ranges_to_next_output=[r for r in trace.peer_rows
                if available and decision_step <= r["step"] <= output_step]))
    return dict(post_fork_preemptions=preemptions, post_fork_commits=commits,
                first_three_recovery_commit_to_next_output=recovery,
                full_episode_position_work=trace.peer_work)


def all_requests_comparison(a, b):
    rows = []
    totals = {side: {k: sum(w[k] for w in trace.request_work.values())
                     for k in ("scheduled", "unique_positions", "repeated_positions")}
              for side, trace in (("host", a), ("other", b))}
    for key in a.requests:
        left, right = a.token_steps[key], b.token_steps[key]
        changes = [i for i, (x, y) in enumerate(zip(left, right)) if x != y]
        if len(left) != len(right):
            changes.extend(range(min(len(left), len(right)), max(len(left), len(right))))
        rows.append(dict(external_id=key, host_finish_step=a.finish[key], other_finish_step=b.finish[key],
            finish_step_equal=a.finish[key] == b.finish[key], host_output_tokens=len(left), other_output_tokens=len(right),
            token_emission_steps_equal=not changes, changed_token_emission_steps=len(changes),
            first_changed_output_index=changes[0] if changes else None,
            native_per_request_step_schedule_equal=a.request_schedule[key] == b.request_schedule[key],
            position_work_other_minus_host={k:b.request_work[key][k]-a.request_work[key][k]
                                           for k in a.request_work[key]}))
    return dict(requests=len(rows), equal_finish_steps=sum(r["finish_step_equal"] for r in rows),
        equal_token_emission_steps=sum(r["token_emission_steps_equal"] for r in rows),
        equal_native_per_request_step_schedules=sum(r["native_per_request_step_schedule_equal"] for r in rows),
        position_work_totals=totals, position_work_other_minus_host={k:totals["other"][k]-totals["host"][k]
                                                                  for k in totals["host"]}, rows=rows)


def analyze(host, other, peer):
    a, b = Trace(host, peer), Trace(other, peer)
    assert a.requests.keys() == b.requests.keys()
    fork = next(i for i, (x, y) in enumerate(zip(a.schedule, b.schedule)) if x != y)
    different_finish = [k for k in a.requests if a.finish[k] != b.finish[k]]
    first_finish = min((min(a.finish[k], b.finish[k]) for k in different_finish), default=None)
    first_ids = [k for k in a.requests if first_finish is not None and
                 (a.finish[k] <= first_finish) != (b.finish[k] <= first_finish)]
    different_lengths = [k for k in a.requests if len(a.requests[k]["output_token_ids"]) != len(b.requests[k]["output_token_ids"])]
    first_length_stop = min((min(a.finish[k], b.finish[k]) for k in different_lengths), default=None)
    length_ids = [k for k in different_lengths if min(a.finish[k], b.finish[k]) == first_length_stop]
    total_count_diff = next((i for i in range(min(len(a.ends), len(b.ends)))
                             if sum(a.count(k, i) for k in a.requests) != sum(b.count(k, i) for k in b.requests)), None)
    snapshots = sorted({fork - 1, fork,
                        *([first_finish - 1, first_finish] if first_finish is not None else []),
                        *([first_length_stop - 1, first_length_stop] if first_length_stop is not None else [])})
    peer_window_end = max(first_finish + 2 if first_finish is not None else fork, fork + 14)
    first_peer_output = min(a.count(peer, fork - 1), b.count(peer, fork - 1))
    return dict(
        host_path=str(a.path), other_path=str(b.path), host_sha256=a.sha256, other_sha256=b.sha256,
        convention="Zero-based formal steps; half-open token-position intervals; outputs mapped by exact step end timestamps.",
        limitations=["Finished-set timing and output length are separate observables.",
                     "Position overlap counts scheduled work, including before the fork; it is not proof of equal KV values or contents.",
                     "Later replay execution and the earlier loss that necessitated it must be distinguished.",
                     "No latency correction, significance test, or fixed-output counterfactual."],
        peer=peer, first_scheduled_work_difference_step=fork,
        all_requests_comparison=all_requests_comparison(a, b),
        peer_output_steps_near_fork=[dict(output_index=i, host_step=a.token_steps[peer][i],
                                         other_step=b.token_steps[peer][i])
            for i in range(first_peer_output, min(first_peer_output+8, len(a.token_steps[peer]), len(b.token_steps[peer])))],
        first_output_content_difference=first_content_difference(a, b),
        first_finished_set_difference=dict(step=first_finish, requests=[request_comparison(a, b, k, first_finish) for k in first_ids]),
        earliest_finish_among_requests_with_different_final_lengths=dict(step=first_length_stop, requests=[request_comparison(a, b, k, first_length_stop) for k in length_ids]),
        first_total_emitted_count_difference_step=total_count_diff,
        snapshots=[dict(host=a.snapshot(s), other=b.snapshot(s)) for s in snapshots if s >= 0],
        peer_early_schedule=dict(host=[x for x in a.peer_rows if fork <= x["step"] <= peer_window_end],
                                 other=[x for x in b.peer_rows if fork <= x["step"] <= peer_window_end]),
        peer_events=dict(host=peer_events(a, peer, fork), other=peer_events(b, peer, fork)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, type=Path, help="Host raw.json")
    parser.add_argument("--other", required=True, type=Path, help="Recompute/headroom raw.json")
    parser.add_argument("--peer", required=True, help="External identity, e.g. measured/E227")
    args = parser.parse_args()
    print(json.dumps(analyze(args.host, args.other, args.peer), indent=2, ensure_ascii=False))
