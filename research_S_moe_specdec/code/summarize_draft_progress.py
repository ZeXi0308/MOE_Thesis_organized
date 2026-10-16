#!/usr/bin/env python3
"""Summarize actual draft availability/progress from complete performance runs.

  python code/summarize_draft_progress.py runs/olmoe_ngram_matrix_r02 \
      --output runs/derived/draft_progress_r02

Only perf-*.json is read from directories. Route-diagnostic populations are
excluded. Every scheduled decode request-step counts, including zero proposals
and drain batches smaller than the nominal B. all_decode, nonterminal and
terminal populations are reported separately. Host/plugin step identity must
match. Old zero-draft terminal omissions are recovered from each finished
request's last output chunk at the exact host-step timestamp and a unique ID
mapping; missing/ambiguous identities remain explicit.

accepted/proposed is a conditional slot ratio, NOT unconditional alpha or the
probability of every token being accepted. 1+accepted is a progress proxy; actual
client-emitted tokens are counted independently where chunk identity is known.
No absent field is silently replaced by a fabricated observation. Summaries
retain every run value; request-steps are not independent system repetitions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import unittest

from plot_results import REGIME_FIELDS, _context, resolve_regime, route_instrumented, write_csv


METRICS = (
    "proposal_positive_fraction", "scheduled_positive_fraction",
    "mean_draft_slots_per_request_step", "mean_scheduled_draft_slots_per_request_step",
    "mean_accepted_drafts_per_request_step", "one_plus_accepted_progress_proxy",
    "actual_emitted_tokens_per_request_step", "accepted_over_proposed_slots_conditional",
    "accepted_over_scheduled_slots_conditional")


def _new_counter() -> Counter:
    return Counter()


def _external_id(internal: str, external_ids: set[str]) -> str | None:
    if internal in external_ids:
        return internal
    # vLLM adds a unique suffix to external request IDs; require one matching
    # external prefix. Do not strip arbitrary hyphen components heuristically.
    matches = [rid for rid in external_ids if internal.startswith(rid + "-")]
    return matches[0] if len(matches) == 1 else None


def _decode_ids(step: dict) -> tuple[list[str] | None, str]:
    if "scheduled_decode_request_ids" in step:
        return list(step["scheduled_decode_request_ids"]), "explicit"
    if step.get("phase") == "decode" and "scheduled_tokens_by_request" in step:
        return list(step["scheduled_tokens_by_request"]), "pure_decode_scheduled_population"
    if step.get("phase") == "mixed" and "decode_request_ids_before" in step:
        before = set(step["decode_request_ids_before"])
        return [rid for rid in step.get("scheduled_tokens_by_request", {}) if rid in before], "explicit_before_intersection"
    if int(step.get("decode_batch", 0)) == 0:
        return [], "no_decode"
    return None, "unavailable"


def _add(counter: Counter, *, available, scheduled, accepted, emitted,
         controller_trim, scheduler_trim, terminal, finish_reason):
    counter["decode_request_steps"] += 1
    for name, value in (("available_draft_slots", available), ("scheduled_draft_slots", scheduled),
                        ("accepted_drafts", accepted), ("emitted_tokens", emitted),
                        ("controller_trimmed_slots", controller_trim), ("scheduler_trimmed_slots", scheduler_trim)):
        if value is None:
            counter[name + "_missing_request_steps"] += 1
        else:
            counter[name] += value
            counter[name + "_observed_request_steps"] += 1
    if available is not None:
        counter["proposal_positive_request_steps"] += int(available > 0)
    if scheduled is not None:
        counter["scheduled_positive_request_steps"] += int(scheduled > 0)
    if accepted is not None and emitted is not None:
        counter["proxy_emission_compared_request_steps"] += 1
        counter["one_plus_accepted_minus_emitted"] += 1 + accepted - emitted
        counter["proxy_emission_mismatch_request_steps"] += int(1 + accepted != emitted)
    if terminal:
        counter["terminal_request_steps"] += 1
        counter["terminal_length_cap_request_steps"] += int(finish_reason == "length")
        counter["terminal_other_reason_request_steps"] += int(finish_reason != "length")


def _metrics(counter: Counter) -> dict:
    n = counter["decode_request_steps"]
    def complete_mean(name):
        return counter[name] / n if n and counter[name + "_missing_request_steps"] == 0 else None
    accepted = complete_mean("accepted_drafts")
    available_ok = n and not counter["available_draft_slots_missing_request_steps"]
    scheduled_ok = n and not counter["scheduled_draft_slots_missing_request_steps"]
    accepted_ok = n and not counter["accepted_drafts_missing_request_steps"]
    result = {"decode_request_steps": n, **dict(counter),
        "proposal_positive_fraction": counter["proposal_positive_request_steps"] / n if available_ok else None,
        "scheduled_positive_fraction": counter["scheduled_positive_request_steps"] / n if scheduled_ok else None,
        "mean_draft_slots_per_request_step": complete_mean("available_draft_slots"),
        "mean_scheduled_draft_slots_per_request_step": complete_mean("scheduled_draft_slots"),
        "mean_accepted_drafts_per_request_step": accepted,
        "one_plus_accepted_progress_proxy": 1 + accepted if accepted is not None else None,
        "actual_emitted_tokens_per_request_step": complete_mean("emitted_tokens"),
        "accepted_over_proposed_slots_conditional": counter["accepted_drafts"] / counter["available_draft_slots"]
            if accepted_ok and available_ok and counter["available_draft_slots"] else None,
        "accepted_over_scheduled_slots_conditional": counter["accepted_drafts"] / counter["scheduled_draft_slots"]
            if accepted_ok and scheduled_ok and counter["scheduled_draft_slots"] else None}
    return result


def analyze_run(obj: dict, source: str, context: dict | None = None) -> tuple[list[dict], dict]:
    context = context or {}
    if route_instrumented(obj, context):
        raise ValueError("route-instrumented/diagnostic run is not the performance population")
    regime = resolve_regime(obj, context)
    populations = {name: _new_counter() for name in ("all_decode", "nonterminal", "terminal", "terminal_status_unknown")}
    diagnostics = Counter()
    actual_batches = Counter()
    wave_rows = []
    total_steps = total_decode_steps = steps_with_terminal = 0
    native_invalid_slots = native_invalid_steps = 0
    for episode in obj.get("episodes", [obj]):
        hosts = episode.get("host_steps", [])
        steps = episode.get("plugin", {}).get("steps", [])
        if len(hosts) != len(steps) or not hosts:
            raise ValueError(f"{episode.get('label')}: host/plugin step count absent or mismatched")
        requests = {r["request_id"]: r for r in episode.get("requests", [])}
        external_ids = set(requests)
        chunks = {rid: {float(c["elapsed_s"]): int(c["tokens"]) for c in r.get("chunks", [])}
                  for rid, r in requests.items()}
        final_times = {rid: float(r["chunks"][-1]["elapsed_s"]) for rid, r in requests.items()
                       if r.get("finished") and r.get("chunks")}
        id_cache = {}
        wave_counter = _new_counter()
        for index, (host, step) in enumerate(zip(hosts, steps)):
            total_steps += 1
            if int(host.get("index", -1)) != index or int(step.get("step", -1)) != index:
                raise ValueError(f"{episode.get('label')}: host/plugin step index mismatch at {index}")
            if step.get("episode") not in (None, episode.get("label")):
                raise ValueError("plugin step has a different wave identity")
            ids, identity_source = _decode_ids(step)
            diagnostics["decode_identity_" + identity_source] += 1
            if ids is None:
                diagnostics["unidentified_decode_request_steps"] += int(step.get("decode_batch", 0))
                continue
            if not ids:
                continue
            if len(ids) != int(step.get("decode_batch", len(ids))):
                raise ValueError("decode identity cardinality differs from native decode_batch")
            total_decode_steps += 1
            actual_batches[len(ids)] += 1
            elapsed = float(host["elapsed_s"]) if "elapsed_s" in host else None
            available_map = step.get("proposed_before_by_request")
            scheduled_map = step.get("scheduled_drafts_by_request")
            accepted_map = step.get("accepted_by_request")
            controller_map = step.get("truncated_by_request")
            censored = set(step.get("ewma_censored_requests", []))
            if "observed_drafts_by_request" not in step:
                diagnostics["native_verified_drafts_per_request_unrecorded_steps"] += 1
            if step.get("phase") == "decode" and "scheduled_drafts" in step and "observed_drafts" in step:
                invalid = int(step["scheduled_drafts"]) - int(step["observed_drafts"])
                if invalid < 0:
                    raise ValueError("native verified draft count exceeds scheduled draft count")
                native_invalid_slots += invalid
                native_invalid_steps += int(invalid > 0)
            terminal_in_step = False
            for rid in ids:
                if rid not in id_cache:
                    id_cache[rid] = _external_id(rid, external_ids)
                external = id_cache[rid]
                runner_known = external is not None and elapsed is not None
                terminal_runner = runner_known and final_times.get(external) == elapsed
                terminal_plugin = rid in censored
                terminal = bool(terminal_runner or terminal_plugin)
                terminal_known = runner_known or terminal_plugin
                if terminal_runner and not terminal_plugin:
                    diagnostics["terminal_request_steps_recovered_from_runner"] += 1
                if not runner_known:
                    diagnostics["unmapped_client_request_steps"] += 1
                terminal_in_step |= terminal
                available = int(available_map[rid]) if available_map is not None and rid in available_map else None
                scheduled = int(scheduled_map.get(rid, 0)) if scheduled_map is not None else None
                accepted = int(accepted_map[rid]) if accepted_map is not None and rid in accepted_map else (0 if scheduled == 0 else None)
                # A positive scheduled draft with no acceptance event is not
                # silently counted as rejected: invalid/terminal clipping may
                # have suppressed native stats, especially in older snapshots.
                emitted = chunks[external].get(elapsed, 0) if runner_known else None
                controller_trim = int(controller_map.get(rid, 0)) if controller_map is not None else None
                scheduler_trim = max(0, available - controller_trim - scheduled) \
                    if None not in (available, controller_trim, scheduled) else None
                if available is not None and available < 0 or scheduled is not None and scheduled < 0:
                    raise ValueError("negative draft count")
                if accepted is not None and (accepted < 0 or scheduled is not None and accepted > scheduled):
                    raise ValueError("invalid accepted draft count")
                finish_reason = requests.get(external, {}).get("finish_reason") if terminal else None
                values = dict(available=available, scheduled=scheduled, accepted=accepted, emitted=emitted,
                              controller_trim=controller_trim, scheduler_trim=scheduler_trim,
                              terminal=terminal, finish_reason=finish_reason)
                _add(populations["all_decode"], **values)
                _add(populations["terminal" if terminal else "nonterminal" if terminal_known else "terminal_status_unknown"], **values)
                _add(wave_counter, **values)
            steps_with_terminal += int(terminal_in_step)
        wave_rows.append({"wave": episode.get("label"), **_metrics(wave_counter)})
    rows = [{**regime, "source": source, "label": obj.get("label"), "batch": int(obj["batch"]),
             "nominal_gamma": int(obj["gamma"]), "proposal_budget": int(obj.get("proposal_budget", obj["gamma"])),
             "mode": obj.get("mode", "unspecified"), "population": name, **_metrics(counter),
             "all_finished": obj.get("all_finished"), "input_request_count": len(obj.get("requests", []))
                or sum(len(e.get("requests", [])) for e in obj.get("episodes", [])),
             "host_steps": total_steps, "decode_host_steps": total_decode_steps,
             "host_steps_with_terminal_decode_requests": steps_with_terminal,
             "native_invalid_slots_pure_decode_all": native_invalid_slots,
             "native_invalid_steps_pure_decode_all": native_invalid_steps,
             "actual_decode_batch_histogram": json.dumps(dict(sorted(actual_batches.items()))),
             "missing_and_identity_diagnostics": json.dumps(dict(diagnostics)),
             "boundary": "all scheduled decode request-steps; 1+accepted is a proxy; conditional slot ratios are not unconditional alpha"}
            for name, counter in populations.items() if counter["decode_request_steps"]]
    return rows, {"source": source, "diagnostics": dict(diagnostics), "waves": wave_rows,
                  "actual_decode_batch_histogram": dict(actual_batches),
                  "native_invalid_slots_boundary": "exact aggregate for pure-decode steps; cannot assign to individual requests when per-request verified counts are absent"}


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    fields = (*REGIME_FIELDS, "batch", "nominal_gamma", "proposal_budget", "mode", "population")
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    output = []
    for key, obs in sorted(groups.items()):
        item = dict(zip(fields, key))
        item.update(n_runs=len(obs), raw_sources=json.dumps([r["source"] for r in obs]),
                    decode_request_steps_raw=json.dumps([r["decode_request_steps"] for r in obs]))
        for metric in METRICS:
            raw = [r[metric] for r in obs]
            known = [v for v in raw if v is not None]
            item[metric + "_raw"] = json.dumps(raw)
            item[metric + "_median"] = statistics.median(known) if known and len(known) == len(raw) else None
            item[metric + "_observed_runs"] = len(known)
        item["aggregation_boundary"] = "run values retained; no confidence interval inferred from request-steps"
        output.append(item)
    return output


def self_test():
    def fixture():
        return {"label": "run", "batch": 2, "gamma": 3, "requests": [], "episodes": [{
            "label": "w", "requests": [
                {"request_id": "a", "finished": True, "finish_reason": "length", "chunks": [{"elapsed_s": 1., "tokens": 2}]},
                {"request_id": "b", "finished": False, "chunks": [{"elapsed_s": 1., "tokens": 1}]}],
            "host_steps": [{"index": 0, "elapsed_s": 1.}], "plugin": {"steps": [{
                "step": 0, "episode": "w", "phase": "decode", "decode_batch": 2,
                "scheduled_tokens_by_request": {"a-1234": 3, "b-1234": 1},
                "proposed_before_by_request": {"a-1234": 3, "b-1234": 0},
                "scheduled_drafts_by_request": {"a-1234": 2}, "accepted_by_request": {"a-1234": 1},
                "truncated_by_request": {}, "scheduled_drafts": 2, "observed_drafts": 2,
                "ewma_censored_requests": []}]}}]}
    class Tests(unittest.TestCase):
        def test_zero_draft_and_terminal_recovery(self):
            rows, details = analyze_run(fixture(), "synthetic")
            all_rows = next(r for r in rows if r["population"] == "all_decode")
            self.assertEqual(all_rows["decode_request_steps"], 2)
            self.assertEqual(all_rows["proposal_positive_fraction"], .5)
            self.assertEqual(all_rows["mean_draft_slots_per_request_step"], 1.5)
            self.assertEqual(all_rows["one_plus_accepted_progress_proxy"], 1.5)
            self.assertEqual(all_rows["actual_emitted_tokens_per_request_step"], 1.5)
            self.assertEqual(next(r for r in rows if r["population"] == "terminal")["scheduler_trimmed_slots"], 1)
            self.assertEqual(details["diagnostics"]["terminal_request_steps_recovered_from_runner"], 1)
        def test_missing_proposals_not_zero_filled(self):
            data = fixture()
            del data["episodes"][0]["plugin"]["steps"][0]["proposed_before_by_request"]
            rows, _ = analyze_run(data, "synthetic")
            self.assertIsNone(rows[0]["proposal_positive_fraction"])
        def test_raw_repeat_values_retained(self):
            rows, _ = analyze_run(fixture(), "synthetic")
            out = summarize(rows + rows)
            self.assertEqual(json.loads(out[0]["one_plus_accepted_progress_proxy_raw"]), [1.5, 1.5])
        def test_misalignment_rejected(self):
            data = fixture()
            data["episodes"][0]["host_steps"] = []
            with self.assertRaises(ValueError):
                analyze_run(data, "synthetic")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.output:
        parser.error("--output is required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    paths = set()
    for item in map(Path, args.inputs):
        if item.is_dir():
            paths.update(p.resolve() for p in item.rglob("perf-*.json"))
        else:
            paths.add(item.resolve())
    runs, details, excluded = [], [], []
    for path in sorted(paths):
        try:
            raw = path.read_bytes()
            obj = json.loads(raw)
            if str(obj.get("label", "")).startswith("warmup"):
                raise ValueError("warmup is not a measured performance run")
            rows, detail = analyze_run(obj, str(path), _context(path))
            detail["source_sha256"] = hashlib.sha256(raw).hexdigest()
            for row in rows:
                row["source_sha256"] = detail["source_sha256"]
            runs.extend(rows)
            details.append(detail)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            excluded.append({"source": str(path), "reason": str(exc)})
    summary = summarize(runs)
    write_csv(output / "draft_progress_runs.csv", runs)
    write_csv(output / "draft_progress_summary.csv", summary)
    (output / "draft_progress.json").write_text(json.dumps({"runs": runs, "summary": summary,
        "run_details": details, "excluded": excluded,
        "evidence_kind": "complete_performance_run_decode_population",
        "conditional_ratio_boundary": "accepted/proposed slot ratios are not unconditional alpha",
        "proxy_boundary": "1+native accepted may differ from emitted tokens on terminal/output-capped steps"}, indent=2))
    print(json.dumps({"output": str(output), "input_runs": len(paths), "analyzed_runs": len(details),
                      "excluded_runs": len(excluded), "summary_rows": len(summary)}))


if __name__ == "__main__":
    main()
