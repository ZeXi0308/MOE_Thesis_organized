#!/usr/bin/env python3
"""CPU-only, cap-only single-donor admission rule; no native adapter or GPU use.

This is a partial CacheOPT-inspired temporal reuse ablation, not CacheOPT.
All current residents must have complete logical reservations. The caller must
establish synchronous, protected, one-output-per-call decoding through cap/EOS.
A later adapter must permit only one outstanding promise and no promise chains.
"""
from dataclasses import asdict, dataclass
import json
from typing import Optional, Sequence

BLOCK_TOKENS = 16
CAPACITY_BLOCKS = 4096
MAX_SEQUENCES = 32
GUARD_TOKENS = 1


def ceil_blocks(tokens: int) -> int:
    return (tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS


@dataclass(frozen=True)
class Resident:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    max_output_tokens: int
    fully_reserved: bool = True
    promised: bool = False

    @property
    def full_blocks(self) -> int:
        return ceil_blocks(self.prompt_tokens + self.max_output_tokens)

    @property
    def horizon(self) -> int:
        return self.max_output_tokens - self.output_tokens


@dataclass(frozen=True)
class WaitingHead:
    request_id: str
    prompt_tokens: int
    max_output_tokens: int

    @property
    def full_blocks(self) -> int:
        return ceil_blocks(self.prompt_tokens + self.max_output_tokens)


@dataclass(frozen=True)
class Decision:
    admitted: bool
    mode: str
    reason: str
    free_blocks: int
    receiver_full_blocks: int
    receiver_initial_blocks: int = 0
    donor_id: Optional[str] = None
    donor_full_blocks: int = 0
    donor_horizon_calls: Optional[int] = None
    transfer_blocks: int = 0
    receiver_runway_tokens: Optional[int] = None
    guard_tokens: int = GUARD_TOKENS


def _validate(residents: Sequence[Resident], head: WaitingHead) -> None:
    ids = [r.request_id for r in residents] + [head.request_id]
    if any(not isinstance(rid, str) or not rid for rid in ids) or len(ids) != len(set(ids)):
        raise ValueError("request IDs must be nonempty and unique")
    for r in [*residents, head]:
        for value in (r.prompt_tokens, r.max_output_tokens):
            if type(value) is not int or value <= 0:
                raise ValueError("prompt and cap must be positive integers")
        if r.prompt_tokens + r.max_output_tokens > 4096:
            raise ValueError("outside the fixed 4096-token model context")
    for r in residents:
        if type(r.output_tokens) is not int or not 0 <= r.output_tokens <= r.max_output_tokens:
            raise ValueError("resident output count outside its known cap")


def choose_admission(
    residents: Sequence[Resident],
    head: WaitingHead,
    *,
    pure_decode_protected_prefix: bool,
    outstanding_promise: bool = False,
) -> Decision:
    """Choose the first feasible unpromised donor in resident input order.

    F = C - sum(ceil((P+M)/16)) is logical free capacity, not physical free KV.
    This function neither mutates state nor records a promise. The caller must
    apply a returned promise atomically and charge the receiver's initial quota.
    While a promise is outstanding this standalone V1 returns HOLD; it must not
    compute new credit by pretending the partial receiver is fully reserved.
    """
    _validate(residents, head)
    full = head.full_blocks
    # A partial receiver cannot be accounted for with the full-reservation sum.
    if outstanding_promise or any(not r.fully_reserved for r in residents):
        return Decision(False, "hold", "outstanding_or_partial_reservation", -1, full)
    free = CAPACITY_BLOCKS - sum(r.full_blocks for r in residents)
    if free < 0:
        raise ValueError("full-reservation ledger already exceeds capacity")
    if not pure_decode_protected_prefix or any(not 0 < r.output_tokens < r.max_output_tokens
                                               for r in residents):
        return Decision(False, "hold", "protected_decode_condition_required", free, full)
    if len(residents) >= MAX_SEQUENCES:
        return Decision(False, "hold", "sequence_cap", free, full)
    if free >= full:
        return Decision(True, "full_bound", "ordinary_full_reservation", free, full,
                        receiver_initial_blocks=full)
    runway = BLOCK_TOKENS * free - head.prompt_tokens
    if runway < 0:
        return Decision(False, "hold", "head_prefill_exceeds_initial_quota", free, full,
                        receiver_runway_tokens=runway)
    shortfall = full - free
    for donor in residents:
        if (not donor.promised and donor.horizon + GUARD_TOKENS <= runway
                and donor.full_blocks >= shortfall):
            return Decision(True, "single_donor", "cap_donor_promise", free, full,
                            receiver_initial_blocks=free, donor_id=donor.request_id,
                            donor_full_blocks=donor.full_blocks,
                            donor_horizon_calls=donor.horizon,
                            transfer_blocks=shortfall, receiver_runway_tokens=runway)
    return Decision(False, "hold", "no_feasible_unpromised_donor", free, full,
                    receiver_runway_tokens=runway)


def exhaustive_token_step_check(
    residents: Sequence[Resident], head: WaitingHead, decision: Decision,
    *, donor_eos_after_calls: Optional[int] = None,
    receiver_eos_after_calls: Optional[int] = None,
) -> dict:
    """Check every integer token step of a conservative mathematical occupancy.

    This is not GPU replay. Receiver prefill is charged immediately and its
    subsequent progress is maximal, with one extra token of lookahead. Existing
    residents progress once per call. A request remains charged at its last
    pre-release step; post-release occupancy is checked separately. Earlier EOS
    parameters are only hypothetical test inputs, never admission inputs.
    """
    if not decision.admitted:
        raise ValueError("an admitted decision is required")
    donor = next((r for r in residents if r.request_id == decision.donor_id), None)
    if donor_eos_after_calls is not None and donor is None:
        raise ValueError("no donor in this decision")
    donor_end = donor.horizon if donor else None
    if donor_eos_after_calls is not None:
        if type(donor_eos_after_calls) is not int or not 1 <= donor_eos_after_calls <= donor.horizon:
            raise ValueError("hypothetical donor EOS must precede its cap horizon")
        donor_end = donor_eos_after_calls
    head_end = head.max_output_tokens
    if receiver_eos_after_calls is not None:
        if type(receiver_eos_after_calls) is not int or not 1 <= receiver_eos_after_calls <= head_end:
            raise ValueError("hypothetical receiver EOS must precede its cap horizon")
        head_end = receiver_eos_after_calls
    horizon = max([head.max_output_tokens, *(r.horizon for r in residents)])
    peak, peak_step, receiver_pre_transfer_peak = 0, None, 0
    for t in range(horizon + 1):
        resident_pre = resident_post = 0
        for r in residents:
            end = donor_end if r.request_id == decision.donor_id else r.horizon
            used = ceil_blocks(min(r.prompt_tokens + r.max_output_tokens,
                                   r.prompt_tokens + r.output_tokens + t))
            resident_pre += used if t <= end else 0
            resident_post += used if t < end else 0
        receiver = ceil_blocks(min(head.prompt_tokens + head.max_output_tokens,
                                   head.prompt_tokens + t + GUARD_TOKENS))
        head_pre = receiver if t <= head_end else 0
        head_post = receiver if t < head_end else 0
        if donor is not None and t <= donor_end:
            receiver_pre_transfer_peak = max(receiver_pre_transfer_peak, head_pre)
            assert head_pre <= decision.receiver_initial_blocks, "initial quota exhausted before transfer"
        before, after = resident_pre + head_pre, resident_post + head_post
        assert max(before, after) <= CAPACITY_BLOCKS, (t, before, after)
        if before > peak:
            peak, peak_step = before, t
    assert decision.receiver_initial_blocks + decision.transfer_blocks == head.full_blocks
    if donor:
        assert decision.transfer_blocks <= donor.full_blocks
        assert sum(r.full_blocks for r in residents if r.request_id != donor.request_id) + head.full_blocks <= CAPACITY_BLOCKS
    return dict(checked_integer_steps=horizon + 1, conservative_peak_blocks=peak,
                peak_step=peak_step, receiver_pre_transfer_peak_blocks=receiver_pre_transfer_peak,
                donor_eos_after_calls=donor_end, receiver_eos_after_calls=head_end)


def global_envelope_check(residents: Sequence[Resident], head: WaitingHead) -> dict:
    """Synthetic comparison using all token steps, not observed execution data."""
    horizon = max((r.horizon for r in residents), default=0)
    peak = max((sum(ceil_blocks(r.prompt_tokens + r.output_tokens + t)
                    for r in residents if t <= r.horizon)
                for t in range(horizon + 1)), default=0)
    return dict(resident_envelope_peak=peak, head_full_blocks=head.full_blocks,
                envelope_plus_head=peak + head.full_blocks,
                accepted=peak + head.full_blocks <= CAPACITY_BLOCKS)


def self_check() -> dict:
    def cohort(early_horizon, late_horizon=None):
        horizons = ([early_horizon] * 16 if late_horizon is None else
                    [early_horizon] * 8 + [late_horizon] * 8)
        return [Resident(f"r{i}", 2976, 1024 - h, 1024) for i, h in enumerate(horizons)]

    # Equality: 4000 reserved blocks, F=96, P=1500, runway=36, H+guard=36.
    residents = cohort(35)
    head = WaitingHead("head", 1500, 1024)
    decision = choose_admission(residents, head, pure_decode_protected_prefix=True)
    assert decision.mode == "single_donor" and decision.donor_id == "r0"
    assert (decision.free_blocks, decision.receiver_full_blocks, decision.transfer_blocks) == (96, 158, 62)
    latest = exhaustive_token_step_check(residents, head, decision)
    assert latest["conservative_peak_blocks"] == 4096
    equality_global = global_envelope_check(residents, head)
    assert not equality_global["accepted"] and equality_global["envelope_plus_head"] == 4158
    # Exhaust all possible donor early-EOS steps, and representative receiver
    # EOS steps. Any additional earlier EOS only removes nonnegative occupancy.
    checks = 0
    for donor_end in range(1, 36):
        for head_end in (1, 2, 16, 512, 1024):
            exhaustive_token_step_check(residents, head, decision,
                                        donor_eos_after_calls=donor_end,
                                        receiver_eos_after_calls=head_end)
            checks += 1
    ordinary = choose_admission(residents[:15], head, pure_decode_protected_prefix=True)
    assert ordinary.mode == "full_bound" and ordinary.donor_id is None
    exhaustive_token_step_check(residents[:15], head, ordinary)
    guard_boundary = choose_admission(residents, WaitingHead("head", 1501, 1024),
                                      pure_decode_protected_prefix=True)
    assert not guard_boundary.admitted and guard_boundary.receiver_runway_tokens == 35
    assert not choose_admission(residents, head, pure_decode_protected_prefix=False).admitted
    assert not choose_admission(residents, head, pure_decode_protected_prefix=True,
                                outstanding_promise=True).admitted
    blocked_first = [Resident(r.request_id, r.prompt_tokens, r.output_tokens,
                              r.max_output_tokens, promised=(i == 0))
                     for i, r in enumerate(residents)]
    assert choose_admission(blocked_first, head, pure_decode_protected_prefix=True).donor_id == "r1"
    examples = []
    for label, early, prompt in (("prefill_does_not_fit", 64, 1600),
                                 ("donor_too_late", 64, 1500),
                                 ("one_token_guard_boundary", 36, 1500)):
        rows = cohort(early, 960)
        waiting = WaitingHead("head", prompt, 1024)
        single = choose_admission(rows, waiting, pure_decode_protected_prefix=True)
        global_result = global_envelope_check(rows, waiting)
        assert not single.admitted and global_result["accepted"]
        examples.append(dict(label=label, residents="8 x (P=2976,M=1024,H=early) + 8 x (P=2976,M=1024,H=960)",
                             early_horizon=early, head_prompt_tokens=prompt,
                             single_donor=asdict(single), global_envelope=global_result))
    return dict(status="CPU_MATHEMATICAL_CHECKS_PASSED_NOT_GPU_EVIDENCE",
                fixed_constants=dict(capacity_blocks=CAPACITY_BLOCKS, block_tokens=BLOCK_TOKENS,
                                     max_sequences=MAX_SEQUENCES, guard_tokens=GUARD_TOKENS),
                equality_case=asdict(decision), equality_latest_completion_check=latest,
                equality_global_envelope=equality_global,
                earlier_eos_scenarios=checks, synthetic_separating_examples=examples)


if __name__ == "__main__":
    print(json.dumps(self_check(), indent=2, sort_keys=True))
