"""KV-feasibility admission: cap the running set by remaining KV, not by latency.

Motivation is the measured capacity cliff in
`outputs/admission_capacity/20260908_kv_pressure_probe_r01`:

  * expert weights hold 12.00 GiB, 93.1% of all parameters, leaving KV 15.00 GiB
    (122,848 tokens at 128 KiB/token);
  * 32 long requests need 131,072 tokens, i.e. 6.7% more than capacity, and both
    repeats of `long-cap32` ended in `CAPACITY_BOUNDARY_STOP` with 0/32 completed;
  * `long-cap16` completed 32/32 in both repeats.

So in this regime admission decides *feasibility*, not just speed. A static cap
can express that only if someone already knows the safe number. This policy
derives it from state the scheduler can read before committing an admission.

The rule is deliberately arithmetic and parameter-poor:

    admit the next queued request only if
        reserved_tokens(running) + worst_case_tokens(candidate) <= usable_tokens

where `usable_tokens = kv_capacity_tokens * (1 - safety_margin)` and
`worst_case_tokens(r) = prompt_tokens(r) + remaining_output_tokens(r)`.

Two properties make this a legal online rule rather than an oracle:

  * It uses only the candidate's own prompt length and the run's fixed output
    length, both known at submission time, plus the current running set. No
    future arrival, no future route, no outcome information.
  * It is conservative by construction: it reserves each request's *final*
    footprint, so a request admitted under the rule can always run to completion
    without eviction. That is why it can hold a feasibility guarantee that a
    latency-feedback rule cannot.

The cost of that guarantee is real and must be reported, not hidden: reserving
worst-case footprint under-uses KV early in a request's life, so the rule will
admit fewer requests than a KV-aware policy that tracked actual block usage.
This module makes no claim that the guarantee is free.
"""
from __future__ import annotations

import math


class KVFeasibilityAdmission:
    """Decide admission from KV reservation arithmetic only."""

    def __init__(self, *, kv_capacity_tokens, output_tokens, safety_margin=0.05,
                 engine_max_seqs=None):
        if type(kv_capacity_tokens) is not int or kv_capacity_tokens < 1:
            raise ValueError("KV capacity must be a positive integer token count")
        if type(output_tokens) is not int or output_tokens < 1:
            raise ValueError("output token count must be a positive integer")
        if not isinstance(safety_margin, (int, float)) or not math.isfinite(safety_margin) \
                or not 0.0 <= safety_margin < 1.0:
            raise ValueError("safety margin must lie in [0, 1)")
        if engine_max_seqs is not None and (type(engine_max_seqs) is not int or engine_max_seqs < 1):
            raise ValueError("engine max seqs must be a positive integer when given")
        self.kv_capacity_tokens = kv_capacity_tokens
        self.output_tokens = output_tokens
        self.safety_margin = float(safety_margin)
        self.engine_max_seqs = engine_max_seqs
        self.usable_tokens = int(kv_capacity_tokens * (1.0 - self.safety_margin))
        self.decisions = []

    def footprint_tokens(self, prompt_tokens):
        """Final KV footprint of a request, known before it is admitted."""
        if type(prompt_tokens) is not int or prompt_tokens < 1:
            raise ValueError("prompt token count must be a positive integer")
        return prompt_tokens + self.output_tokens

    def reserved_tokens(self, running_prompt_tokens):
        return sum(self.footprint_tokens(p) for p in running_prompt_tokens)

    def safe_cap_for_uniform_prompt(self, prompt_tokens):
        """How many identical requests fit. Used for reporting, not for deciding."""
        per_request = self.footprint_tokens(prompt_tokens)
        fits = self.usable_tokens // per_request
        if self.engine_max_seqs is not None:
            fits = min(fits, self.engine_max_seqs)
        return max(0, int(fits))

    def decide(self, *, candidate_prompt_tokens, running_prompt_tokens, waiting, now_s):
        """Admit or hold one candidate, using only present state."""
        if not math.isfinite(now_s) or waiting < 0:
            raise ValueError("invalid present scheduler state")
        running = list(running_prompt_tokens)
        reserved = self.reserved_tokens(running)
        need = self.footprint_tokens(candidate_prompt_tokens)
        would_reserve = reserved + need
        engine_blocked = (self.engine_max_seqs is not None
                          and len(running) >= self.engine_max_seqs)
        admit = (not engine_blocked) and would_reserve <= self.usable_tokens
        if engine_blocked:
            reason = "engine_max_seqs"
        elif admit:
            reason = "fits_reservation"
        else:
            reason = "would_exceed_usable_kv"
        row = dict(decision_index=len(self.decisions), now_s=now_s, admit=admit, reason=reason,
                   n_running=len(running), waiting=waiting,
                   reserved_tokens=reserved, candidate_footprint_tokens=need,
                   would_reserve_tokens=would_reserve, usable_tokens=self.usable_tokens,
                   kv_capacity_tokens=self.kv_capacity_tokens,
                   reservation_utilisation=would_reserve / self.usable_tokens
                   if self.usable_tokens else None)
        self.decisions.append(row)
        return row

    def summary(self):
        admitted = [d for d in self.decisions if d["admit"]]
        blocked = [d for d in self.decisions if not d["admit"]]
        return dict(
            n_decisions=len(self.decisions), n_admitted=len(admitted), n_blocked=len(blocked),
            blocked_by_kv=sum(d["reason"] == "would_exceed_usable_kv" for d in blocked),
            blocked_by_engine=sum(d["reason"] == "engine_max_seqs" for d in blocked),
            max_reserved_tokens=max((d["reserved_tokens"] for d in self.decisions), default=0),
            max_reservation_utilisation=max(
                (d["reservation_utilisation"] for d in self.decisions
                 if d["reservation_utilisation"] is not None), default=None),
            usable_tokens=self.usable_tokens, kv_capacity_tokens=self.kv_capacity_tokens,
            safety_margin=self.safety_margin,
            semantics=("reservation is worst-case final footprint per admitted request, so an "
                       "admitted request can always complete without eviction; this deliberately "
                       "under-uses KV early in each request's life"))


def kv_capacity_tokens_from_engine(engine):
    """Read the allocator's KV token capacity; never estimate it."""
    cache = engine.vllm_config.cache_config
    blocks = getattr(cache, "num_gpu_blocks", None)
    block_size = getattr(cache, "block_size", None)
    if not blocks or not block_size:
        raise ValueError("engine did not report KV block capacity")
    return int(blocks) * int(block_size)
