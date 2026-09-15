"""CPU-only candidate decision, evaluated after existing commit validation.

Does not install a runtime hook. Caller retains target promotion and the existing
full-history recovery guard; only optional victim preemption may be skipped.
"""

def choose_commit_action(existing_reason, free_blocks, target_remaining_blocks):
    if free_blocks < 0 or target_remaining_blocks < 0:
        raise ValueError('invalid block accounting')
    if existing_reason != 'READY':
        return 'CANCEL_EXISTING_REASON'
    if free_blocks >= target_remaining_blocks:
        return 'RESUME_WITHOUT_VICTIM'
    return 'COMMIT_EXISTING_SWAP'
