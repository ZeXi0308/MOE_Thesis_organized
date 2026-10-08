"""CPU contract for staged save/preempt. No engine hooks or transfer execution.

StoreEvidence must come from a native job covering this prefix and ownership;
constructing it here is not proof that a real store has been registered.
"""
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class RequestState:
    request_id: str
    computed: int
    prompt: int
    output: int
    max_output: int
    status: str
    blocks: tuple[int, ...]

    @property
    def pure_decode(self):
        return self.output > 0 and self.computed == self.prompt + self.output - 1

    @property
    def remaining_blocks(self):
        return max(0, ceil((self.prompt + self.output) / 16) - len(self.blocks))


@dataclass(frozen=True)
class Plan:
    step: int
    victim: RequestState
    target: RequestState
    saved_tokens: int
    source_blocks: tuple[int, ...]


@dataclass(frozen=True)
class StoreEvidence:
    request_id: str
    prefix_tokens: int
    source_blocks: tuple[int, ...]
    registered_job_ids: tuple[int, ...]


def prepare(step, victim, target, free_blocks):
    if victim.request_id == target.request_id:
        raise ValueError('victim and original absence target must differ')
    if victim.status != 'RUNNING' or not victim.pure_decode:
        raise ValueError('victim must be running pure decode')
    if victim.output + 1 >= victim.max_output:
        raise ValueError('victim will finish on the preparation step')
    if target.status != 'PREEMPTED':
        raise ValueError('target must be waiting for recovery, not an in-flight load')
    if free_blocks < 0 or free_blocks + len(victim.blocks) < target.remaining_blocks:
        raise ValueError('victim cannot fund target full history')
    # Only already materialized blocks are included. The next decode position
    # may need a newly allocated block that cannot be named at prepare time.
    saved = victim.computed // 16 * 16
    source = victim.blocks[:saved // 16]
    if not saved or len(source) != saved // 16 or len(set(source)) != len(source):
        raise ValueError('invalid complete prefix ownership')
    return Plan(step, victim, target, saved, source)


def commit_reason(plan, step, victim, target, free_blocks, store, *, save_enabled=True):
    """Return READY or a cancellation reason; never mutate native state."""
    if step != plan.step + 1:
        return 'CANCEL_STEP_CHANGED'
    if victim is None or target is None:
        return 'CANCEL_REQUEST_FINISHED'
    if (victim.request_id != plan.victim.request_id or victim.status != 'RUNNING'
            or not victim.pure_decode or victim.computed != plan.victim.computed + 1
            or victim.output != plan.victim.output + 1):
        return 'CANCEL_VICTIM_CHANGED'
    if target != plan.target:
        return 'CANCEL_TARGET_CHANGED'
    if victim.blocks[:len(plan.source_blocks)] != plan.source_blocks:
        return 'CANCEL_BLOCK_OWNERSHIP_CHANGED'
    if free_blocks < 0 or free_blocks + len(victim.blocks) < target.remaining_blocks:
        return 'CANCEL_TARGET_UNFUNDED'
    if save_enabled and (store is None or not store.registered_job_ids
                         or store.request_id != victim.request_id
                         or store.prefix_tokens < plan.saved_tokens
                         or store.source_blocks[:len(plan.source_blocks)] != plan.source_blocks):
        return 'CANCEL_STORE_NOT_REGISTERED'
    return 'READY'


def recovery_guard(target, free_blocks, other_growth_blocks, *, load_completion_visible=False):
    """Outstanding target growth is reserved even while its load is pending."""
    if free_blocks < 0 or other_growth_blocks < 0:
        raise ValueError('negative resource count')
    if target.status not in ('PREEMPTED','RUNNING','WAITING_FOR_REMOTE_KVS'):
        raise ValueError('unsupported recovery status')
    pending = target.status == 'WAITING_FOR_REMOTE_KVS' and not load_completion_visible
    reserve = target.remaining_blocks
    return dict(target_schedulable=not pending,
                other_growth_allowed=other_growth_blocks <= free_blocks-reserve,
                reserved_free_blocks=reserve,
                allocated_target_blocks=len(target.blocks))
