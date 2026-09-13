"""CPU accounting primitives, not a fitted policy or a serving simulator.

Future route inputs must be estimates from a completed-history cutoff. Exact
expert sets are legal only after the router, or for labelled offline diagnosis.
No function infers GPU time from logical bytes or changes native scheduling.
"""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class RequestState:
    request_id: str
    prompt_tokens: int
    computed_tokens: int
    output_tokens: int
    max_output_tokens: int
    phase: str  # waiting / prefill / decode / recovering / complete
    arrival_s: float = 0.0

    def __post_init__(self):
        if self.phase not in {'waiting', 'prefill', 'decode', 'recovering', 'complete'}:
            raise ValueError('unknown request phase')
        counts = (self.prompt_tokens, self.computed_tokens, self.output_tokens, self.max_output_tokens)
        if any(type(n) is not int or n < 0 for n in counts):
            raise ValueError('token counts must be nonnegative integers')
        if not math.isfinite(self.arrival_s) or self.arrival_s < 0:
            raise ValueError('invalid arrival time')
        if (self.output_tokens > self.max_output_tokens
                or self.computed_tokens > self.prompt_tokens + max(0, self.output_tokens-1)):
            raise ValueError('inconsistent computed/output token state')
        if self.phase == 'waiting' and (self.computed_tokens or self.output_tokens):
            raise ValueError('waiting means never executed')
        if self.phase == 'decode' and (not self.output_tokens or self.recovery_tokens):
            raise ValueError('decode requires a fully computed output prefix')

    def blocks(self, block_size):
        if block_size <= 0:
            raise ValueError('positive block size required')
        return math.ceil(self.computed_tokens / block_size) if self.phase != 'complete' else 0

    @property
    def recovery_tokens(self):
        # The last emitted token need not have been forwarded yet.
        return max(0, self.prompt_tokens + max(0, self.output_tokens - 1) - self.computed_tokens)


def apply_event(request, *, computed_delta=0, returned_tokens=0, event='step'):
    """Apply supplied native events (or explicit hypothetical scenarios).

    Preemption frees KV, retains generated output and enters recovery. Completion
    frees blocks. Lowering a cap is deliberately absent: it is not a KV event.
    """
    if event in {'preempt', 'complete'} and (request.phase in {'waiting', 'complete'}
                                           or computed_delta or returned_tokens):
        raise ValueError('invalid lifecycle transition or combined event')
    if event == 'preempt':
        return replace(request, computed_tokens=0, phase='recovering')
    if event == 'complete':
        return replace(request, phase='complete')
    if event != 'step' or any(type(n) is not int or n < 0 for n in (computed_delta, returned_tokens)):
        raise ValueError('invalid event')
    if request.phase == 'complete':
        raise ValueError('completed request cannot execute')
    if not computed_delta and not returned_tokens:
        return request
    outputs = request.output_tokens + returned_tokens
    computed = request.computed_tokens + computed_delta
    if outputs > request.max_output_tokens or computed > request.prompt_tokens + max(0, outputs-1):
        raise ValueError('work exceeds supplied request/token state')
    if returned_tokens and computed < request.prompt_tokens + outputs-1:
        raise ValueError('output returned before its required computation')
    phase = 'decode' if outputs and computed >= request.prompt_tokens + outputs-1 else 'prefill'
    if request.phase == 'recovering' and phase == 'prefill':
        phase = 'recovering'
    return replace(request, computed_tokens=computed, output_tokens=outputs, phase=phase)


def admission_slots(requests, target_cap, now_s):
    """FCFS by (arrival, request_id); existing decode/recovery is never truncated.

    A native adapter must preserve native enqueue order for equal-arrival ties;
    this deterministic CPU convention is not a claim of native tie equivalence.
    """
    requests = list(requests)
    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError('duplicate request identity')
    if type(target_cap) is not int or target_cap < 0 or not math.isfinite(now_s):
        raise ValueError('invalid admission state')
    active = sum(r.phase not in ('waiting', 'complete') for r in requests)
    waiting = [r.request_id for r in sorted(requests, key=lambda r: (r.arrival_s, r.request_id))
               if r.phase == 'waiting' and r.arrival_s <= now_s]
    return waiting[:max(0, target_cap-active)]


def layer_transfer_bytes(required_by_request, resident, weight_bytes):
    """Known current-layer unique misses, excluding capacity-induced reloads."""
    union = set().union(*(set(experts) for experts in required_by_request))
    missing = union - set(resident)
    return sum(weight_bytes[e] for e in missing)


def independent_expected_bytes(probabilities, resident, weight_bytes):
    """Estimated unique miss bytes under explicitly independent token routes.

    Each row supplies marginal inclusion probabilities, not post-action routes.
    Within-token top-k dependence is compatible; across-token independence is an
    analytical baseline, never a measured assertion about real MoE traffic.
    This is NOT an online forecast interface: it neither accepts nor verifies
    a history cutoff or action provenance. A future adapter must provide those.
    """
    miss = 0.0
    for expert, size in weight_bytes.items():
        if expert in resident:
            continue
        absent = 1.0
        for row in probabilities:
            prob = row.get(expert, 0.0)
            if not math.isfinite(prob) or not 0 <= prob <= 1:
                raise ValueError('invalid route inclusion probability')
            absent *= 1-prob
        miss += size*(1-absent)
    return miss


def uniform_union(experts, top_k, tokens):
    if experts <= 0 or not 0 <= top_k <= experts or tokens < 0:
        raise ValueError('invalid uniform routing dimensions')
    return experts * (1-(1-top_k/experts)**tokens)


def step_time_s(*, nontransfer_s, exposed_transfer_s, scheduler_s):
    """Use mutually exclusive costs; exposed time must already exclude overlap."""
    if (not all(math.isfinite(v) for v in (nontransfer_s, exposed_transfer_s, scheduler_s))
            or min(nontransfer_s, exposed_transfer_s, scheduler_s) < 0):
        raise ValueError('negative cost')
    return nontransfer_s + exposed_transfer_s + scheduler_s
