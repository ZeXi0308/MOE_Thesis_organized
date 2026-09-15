"""Attach actual post-reorder CPU row identities to a synchronous native pager."""
from numbers import Integral


def install_row_context(runner, pager):
    """Install after model initialization; return a callable restoring the hook."""
    if (runner.use_async_scheduling or runner.num_spec_tokens
            or runner.parallel_config.use_ubatching):
        raise ValueError("row context requires synchronous non-speculative non-ubatched execution")
    runtime = getattr(pager, "_runtime", pager)
    if runtime is None:
        raise ValueError("pager must be initialized")
    original = runner._prepare_inputs
    had_instance = "_prepare_inputs" in vars(runner)

    def prepare(scheduler_output, num_scheduled_tokens):
        result = original(scheduler_output, num_scheduled_tokens)
        batch = runner.input_batch
        ids = list(batch.req_ids)
        counts = list(num_scheduled_tokens)
        starts = list(batch.num_computed_tokens_cpu[:batch.num_reqs])
        total = scheduler_output.total_num_scheduled_tokens
        integers = lambda xs: all(isinstance(x, Integral) and not isinstance(x, bool) for x in xs)
        if (len(ids) != batch.num_reqs or len(counts) != len(ids) or len(starts) != len(ids)
                or any(not isinstance(rid, str) or not rid for rid in ids) or len(set(ids)) != len(ids)
                or not integers([total, *counts, *starts]) or any(c <= 0 for c in counts)
                or any(s < 0 for s in starts) or sum(counts) != total
                or set(ids) != set(scheduler_output.num_scheduled_tokens)
                or any(c != scheduler_output.num_scheduled_tokens[rid] for rid, c in zip(ids, counts))):
            raise ValueError("invalid actual request/count/position metadata")
        indices = list(runner.req_indices.np[:total])
        expected_indices = [i for i, count in enumerate(counts) for _ in range(count)]
        if not integers(indices) or indices != expected_indices:
            raise ValueError("actual input row indices do not match post-reorder request counts")
        rows = [dict(internal_request_id=rid, computed_position=int(start + offset))
                for rid, count, start in zip(ids, counts, starts) for offset in range(count)]
        context = dict(runtime.context)
        context.update(rows=rows, row_request_order_verified=True,
                       valid_row_start=0, valid_row_stop=int(total),
                       row_scope="unpadded actual input rows; any rows beyond valid_row_stop are unmapped padding")
        pager.set_context(**context)
        return result

    runner._prepare_inputs = prepare

    def uninstall():
        if getattr(runner, "_prepare_inputs", None) is prepare:
            if had_instance:
                runner._prepare_inputs = original
            else:
                del runner._prepare_inputs

    return uninstall
