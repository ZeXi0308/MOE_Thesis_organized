"""One real mapped kernel per call; aggregate CPU interface evidence, no tensor readback."""
import hashlib
import json


def call_identity_bytes(record):
    key = [record['call_id'], record['layer_name'], record['context']['step_id'], record['rows']]
    return (json.dumps(key, separators=(',', ':')) + '\n').encode()


class LogicalExecution:
    def __init__(self, runtime, module, report):
        self.runtime, self.module, self.report = runtime, module, report
        self.original, self.assignment = runtime.kernel, module._prepare_expert_assignment
        self.hashes = {}
        runtime.kernel = self.kernel

    def kernel(self, **kwargs):
        runtime = self.runtime; record = runtime.records[-1]; context = record['context']
        phase = context['phase']
        stats = self.report['phases'].setdefault(phase, dict(calls=0, completed_calls=0, failed_calls=0,
            helper_calls=0, token_rows=0, helper_token_rows=0, rows_histogram={},
            helper_parameters=dict(global_num_experts=64, map_shape=[64], top_k=8,
                incoming_ignore_invalid_experts=True, effective_ignore_invalid_experts=False), assignment_restored=True))
        self.hashes.setdefault(phase, hashlib.sha256()).update(call_identity_bytes(record))
        stats['calls'] += 1; stats['token_rows'] += record['rows']
        width = str(record['rows']); stats['rows_histogram'][width] = stats['rows_histogram'].get(width, 0) + 1
        count = 0
        def assignment(ids, config, num_tokens, top_k_num, global_num_experts, expert_map, **options):
            nonlocal count
            count += 1; stats['helper_calls'] += 1; stats['helper_token_rows'] += num_tokens
            if (count != 1 or global_num_experts != 64 or expert_map.shape != (64,) or top_k_num != 8
                    or num_tokens != record['rows'] or options.get('ignore_invalid_experts') is not True):
                raise RuntimeError('logical assignment interface mismatch')
            options['ignore_invalid_experts'] = False
            return self.assignment(ids, config, num_tokens, top_k_num, global_num_experts, expert_map, **options)
        try:
            if (runtime.shared_mode != 'oneshot' or kwargs.get('global_num_experts') != 384
                    or kwargs['expert_map'].shape != (384,) or kwargs['hidden_states'].shape[0] != record['rows']
                    or any(kwargs[k] is not pool for k, pool in zip(('w1', 'w2'), runtime.shared_weights))
                    or any(pool.shape[0] != 384 for pool in runtime.shared_weights)):
                raise RuntimeError('logical execution requires actual oneshot E384 inputs')
            if self.module._prepare_expert_assignment is not self.assignment:
                raise RuntimeError('assignment helper was concurrently replaced')
            self.module._prepare_expert_assignment = assignment
            actual = self.original(**dict(kwargs, global_num_experts=64, expert_map=kwargs['expert_map'][:64]))
            if count != 1: raise RuntimeError('native kernel did not enter assignment helper exactly once')
            stats['completed_calls'] += 1
            return actual
        except BaseException as exc:
            stats['failed_calls'] += 1; stats['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            self.module._prepare_expert_assignment = self.assignment
            stats['assignment_restored'] &= self.module._prepare_expert_assignment is self.assignment

    def close(self):
        self.runtime.kernel = self.original
        for phase, value in self.hashes.items(): self.report['phases'][phase]['call_identity_sha256'] = value.hexdigest()
        self.report['hooks_restored'] = (self.runtime.kernel is self.original
            and self.module._prepare_expert_assignment is self.assignment)
