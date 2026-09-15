"""Request-level accounting for a finite, fully retained arrival episode.

Latency samples include every observed request/token, even if the request later
fails. SLO pass and completion throughput require status == 'completed'.
"""

import math


def summarize_episode_requests(rows, *, observation_end_s, ttft_slo_s, tpot_slo_s):
    """Retain the planned cohort without inventing future arrivals after an abort."""
    end = _finite(observation_end_s, "observation_end_s")
    rows = list(rows)
    if len({r["request_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate planned request identity")
    arrived, not_yet = [], []
    for row in rows:
        if _finite(row["arrival_s"], "arrival_s") <= end:
            arrived.append(row)
        else:
            if row.get("admission_s") is not None or row["token_times_s"] or row.get("completion_s") is not None:
                raise ValueError("a future request has already executed")
            not_yet.append(row["request_id"])
    result = summarize_requests(arrived, observation_end_s=end,
                                ttft_slo_s=ttft_slo_s, tpot_slo_s=tpot_slo_s)
    result.update(n_planned=len(rows), n_not_yet_arrived=len(not_yet),
                  not_yet_arrived_request_ids=not_yet,
                  denominator_scope="all_arrived_requests_at_actual_observation_end")
    return result


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _distribution(values):
    ordered = sorted(values)

    def percentile(q):
        if not ordered:
            return None
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "n": len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "small_sample": len(ordered) < 100,
        "small_sample_threshold": 100,
        "percentile_method": "linear_interpolation",
    }


def summarize_requests(rows, *, observation_end_s, ttft_slo_s, tpot_slo_s):
    """Summarize all arrivals; never silently drop failed/unfinished requests.

    Time is in seconds on one shared clock. Observation ends after every recorded
    arrival, admission, token and completion. For a nonempty zero-duration episode,
    rates are undefined (None). Empty input has zero attainment and zero rates.
    Completed single-token requests have no TPOT and pass its check vacuously.
    """
    end = _finite(observation_end_s, "observation_end_s")
    ttft_limit = _finite(ttft_slo_s, "ttft_slo_s")
    tpot_limit = _finite(tpot_slo_s, "tpot_slo_s")
    if ttft_limit < 0 or tpot_limit < 0:
        raise ValueError("SLO limits must be nonnegative")
    derived, seen, arrivals, ttfts, tpots, itls = [], set(), [], [], [], []
    counts = {"completed": 0, "failed": 0, "unfinished": 0}
    n_pass = n_vacuous = 0
    for row in rows:
        request_id, status = row["request_id"], row["status"]
        if request_id in seen:
            raise ValueError(f"duplicate request_id: {request_id}")
        seen.add(request_id)
        if status not in counts:
            raise ValueError(f"unknown request status: {status}")
        arrival = _finite(row["arrival_s"], "arrival_s")
        admission = row.get("admission_s")
        admission = None if admission is None else _finite(admission, "admission_s")
        completion = row.get("completion_s")
        completion = None if completion is None else _finite(completion, "completion_s")
        tokens = [_finite(t, "token_times_s") for t in row["token_times_s"]]
        if len(row["output_token_ids"]) != len(tokens):
            raise ValueError("output token IDs and timestamps must align")
        if arrival > end or any(t > end for t in tokens):
            raise ValueError("observation_end_s does not cover arrivals/tokens")
        if admission is not None and not arrival <= admission <= end:
            raise ValueError("admission must be between arrival and observation end")
        if completion is not None and not arrival <= completion <= end:
            raise ValueError("completion must be between arrival and observation end")
        start = arrival if admission is None else admission
        if tokens and (tokens[0] < start or any(b < a for a, b in zip(tokens, tokens[1:]))):
            raise ValueError("token times must be monotonic and after arrival/admission")
        if completion is not None and tokens and completion < tokens[-1]:
            raise ValueError("completion precedes the last token")
        if status == "completed" and completion is None:
            raise ValueError("completed requests need completion_s")
        intervals = [b - a for a, b in zip(tokens, tokens[1:])]
        ttft = tokens[0] - arrival if tokens else None
        tpot = (tokens[-1] - tokens[0]) / (len(tokens) - 1) if len(tokens) >= 2 else None
        vacuous = status == "completed" and len(tokens) == 1
        ttft_pass = ttft is not None and ttft <= ttft_limit
        tpot_pass = (tpot <= tpot_limit) if tpot is not None else vacuous
        slo_pass = status == "completed" and ttft_pass and tpot_pass
        derived.append({
            "request_id": request_id,
            "status": status,
            "n_output_tokens": len(tokens),
            "queue_s": None if admission is None else admission - arrival,
            "ttft_s": ttft,
            "tpot_s": tpot,
            "itl_s": intervals,
            "request_latency_s": None if completion is None else completion - arrival,
            "ttft_pass": ttft_pass,
            "tpot_pass": tpot_pass,
            "tpot_vacuous": vacuous,
            "slo_pass": slo_pass,
        })
        counts[status] += 1
        arrivals.append(arrival)
        n_pass += int(slo_pass)
        n_vacuous += int(vacuous)
        if ttft is not None:
            ttfts.append(ttft)
        if tpot is not None:
            tpots.append(tpot)
        itls.extend(intervals)
    n = len(derived)
    duration = end - min(arrivals) if n else 0.0

    def rate(numerator):
        return numerator / duration if duration > 0 else (0.0 if not n else None)

    return {
        "n_arrived": n,
        "n_completed": counts["completed"],
        "n_failed": counts["failed"],
        "n_unfinished": counts["unfinished"],
        "n_slo_pass": n_pass,
        "n_completed_tpot_vacuous": n_vacuous,
        "slo_attainment": n_pass / n if n else 0.0,
        "goodput_rps": rate(n_pass),
        "throughput_rps": rate(counts["completed"]),
        "observation_duration_s": duration,
        "latency_population": "all_observed_requests_and_tokens",
        "latency_s": {"ttft": _distribution(ttfts), "tpot": _distribution(tpots), "itl": _distribution(itls)},
        "per_request": derived,
    }
