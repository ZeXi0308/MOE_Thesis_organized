"""Aggregate prompt-token cap. Native allocation/order/precision are unchanged."""
import inspect
import textwrap
import types
import hashlib


def install(scheduler):
    assert scheduler.policy.name == 'FCFS'
    assert scheduler.connector is None
    assert scheduler.num_spec_tokens == 0
    original = scheduler.schedule
    source = textwrap.dedent(inspect.getsource(type(scheduler).schedule))
    old = source
    def replace(a, b, count=1):
        nonlocal source
        assert source.count('\n'+a) == count, (a, source.count('\n'+a))
        source = source.replace('\n'+a, '\n'+b)
    replace('    token_budget = self.max_num_scheduled_tokens\n',
            '    token_budget = self.max_num_scheduled_tokens\n    d_prefill_left = self.d_prefill_budget\n')
    replace('        num_new_tokens = min(num_new_tokens, token_budget)\n',
            '        num_new_tokens = min(num_new_tokens, token_budget)\n'
            '        if request.num_prompt_tokens - request.num_computed_tokens > d_prefill_left:\n'
            '            num_new_tokens = min(num_new_tokens, d_prefill_left)\n')
    replace('                num_new_tokens = min(num_new_tokens, token_budget)\n',
            '                num_new_tokens = min(num_new_tokens, token_budget)\n'
            '                if request.num_prompt_tokens - num_computed_tokens > d_prefill_left:\n'
            '                    num_new_tokens = min(num_new_tokens, d_prefill_left)\n'
            '                    if num_new_tokens == 0:\n'
            '                        break\n')
    replace('        token_budget -= num_new_tokens\n',
            '        token_budget -= num_new_tokens\n'
            '        d_prefill_left -= min(num_new_tokens, max(0, request.num_prompt_tokens - request.num_computed_tokens))\n')
    replace('            token_budget -= num_new_tokens\n',
            '            token_budget -= num_new_tokens\n'
            '            d_prefill_left -= min(num_new_tokens, max(0, request.num_prompt_tokens - num_computed_tokens))\n')
    namespace = {}
    exec(compile(source, '<d-prefill-only-scheduler>', 'exec'), original.__func__.__globals__, namespace)
    scheduler.schedule = types.MethodType(namespace['schedule'], scheduler)
    scheduler.d_prefill_budget = 512
    return dict(native_sha256=hashlib.sha256(old.encode()).hexdigest(), patched_sha256=hashlib.sha256(source.encode()).hexdigest(), source=source)


class Policy:
    def __init__(self, name, target_ms=60):
        self.name, self.target_ms = name, target_ms
        self.levels = [256, 512, 1024, 2048]
        self.index = 2
        self.recent_ms = None
        self.mixed_steps = 0

    def choose(self, ndecode):
        if self.name == 'elapsed_replay':
            # OLD development trace switch counts, frozen before this ablation.
            if ndecode == 0:
                return 2048
            m = self.mixed_steps
            if m == 0:
                return 1024
            if m < 9:
                return 2048
            if m < 73:
                return 1024
            if m < 113:
                return 512
            return 256
        if self.name.startswith('fixed'):
            return int(self.name[5:])
        if self.name == 'decode':
            return 2048 if ndecode < 8 else 512
        if self.name == 'decode_v2':
            # Frozen from the four OLD high-load feedback traces only.
            return 2048 if ndecode < 24 else (1024 if ndecode < 35 else 256)
        if self.name == 'feedback':
            if ndecode == 0:
                return 2048
            return self.levels[self.index]
        raise ValueError(self.name)

    def observe(self, elapsed_s, prefill, decode):
        if self.name == 'elapsed_replay':
            if prefill and decode:
                self.mixed_steps += 1
            return
        if prefill and decode:
            self.recent_ms = elapsed_s * 1000
            if self.recent_ms > self.target_ms:
                self.index = max(0, self.index-1)
            elif self.recent_ms < self.target_ms * .7:
                self.index = min(len(self.levels)-1, self.index+1)
