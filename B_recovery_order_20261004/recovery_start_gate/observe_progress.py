"""Same frozen start gate, with a bounded CPU snapshot at its target boundary."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import time

FROZEN_SHA = 'aa6a3a427d44668f41c29af3bb1b2d176be112b81e68dc6275fa8632d94b3d2b'
path = Path(__file__).with_name('start_gate.py')
if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_SHA:
    raise RuntimeError('Frozen start gate changed')
spec = importlib.util.spec_from_file_location('_progress_frozen_start_gate', path)
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
_old_instrument, _old_decision = G._instrument, G._decision

COLUMNS = ('request', 'status', 'history_tokens', 'held_gpu_blocks', 'computed_tokens',
           'num_in_flight_tokens', 'scheduled_tokens_this_step', 'is_running',
           'is_inflight', 'registry_identity', 'plain_unshared_blocks')


def _instrument(tree):
    tree = _old_instrument(tree)
    count = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == G.H.CALLBACK and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in ('decision', 'break_executed')):
            node.args.append(ast.Name(id='num_scheduled_tokens', ctx=ast.Load()))
            count += 1
    if count != 2:
        raise RuntimeError('Expected exactly two native allocation-boundary callbacks')
    return ast.fix_missing_locations(tree)


def _snapshot(current, manager, single, target, scheduled, reserved, now):
    begin = now()
    out = dict(host_perf_s=begin, clock='host perf_counter', status='UNKNOWN', reasons=[],
               native_reserved_blocks=reserved, columns=list(COLUMNS), requests=[])
    reasons = out['reasons']
    try:
        running, inflight = list(current.running), list(current._inflight_prefills)
        running_ids, inflight_ids = {r.request_id for r in running}, {r.request_id for r in inflight}
        if len(running_ids) != len(running) or len(inflight_ids) != len(inflight):
            reasons.append('DUPLICATE_MEMBERSHIP')
        objects = {}
        for req in [*running, *inflight]:
            previous = objects.setdefault(req.request_id, req)
            if previous is not req:
                reasons.append('REQUEST_IDENTITY_CONFLICT')
        if target.request_id in objects and objects[target.request_id] is not target:
            reasons.append('TARGET_IDENTITY_CONFLICT')
        out.update(free_gpu_blocks=manager.block_pool.get_num_free_blocks(),
                   block_size=single.block_size, max_model_len=manager.max_model_len,
                   running_ids=sorted(running_ids), inflight_ids=sorted(inflight_ids))
        if (type(reserved) is not int or reserved < 0
                or type(out['free_gpu_blocks']) is not int or out['free_gpu_blocks'] < 0):
            reasons.append('UNKNOWN_NATIVE_COUNTS')
        if not isinstance(scheduled, dict):
            reasons.append('UNKNOWN_CURRENT_SCHEDULED_MAP')
        seen_blocks = set()

        def read(req):
            rid = req.request_id
            blocks = single.req_to_blocks.get(rid, ())
            plain = True
            for block in blocks:
                bid = getattr(block, 'block_id', None)
                if (getattr(block, 'is_null', None) is not False
                        or getattr(block, 'ref_cnt', None) != 1 or type(bid) is not int
                        or bid in seen_blocks):
                    plain = False
                if type(bid) is int:
                    seen_blocks.add(bid)
            identity = current.requests.get(rid) is req
            n, c, i = req.num_tokens, req.num_computed_tokens, req.num_in_flight_tokens
            q = scheduled.get(rid, 0) if isinstance(scheduled, dict) else None
            if not identity:
                reasons.append('REQUEST_REGISTRY_IDENTITY_UNKNOWN:' + rid)
            if not plain:
                reasons.append('NONPLAIN_OR_DUPLICATE_BLOCKS:' + rid)
            if any(type(v) is not int or v < 0 for v in (n, c, i, q)):
                reasons.append('UNKNOWN_REQUEST_COUNTS:' + rid)
            elif i > c or c + q > n or c + q > len(blocks) * single.block_size:
                reasons.append('UNSUPPORTED_TOKEN_BOUNDARY:' + rid)
            if (getattr(req, 'has_encoder_inputs', None) is not False
                    or getattr(req, 'num_output_placeholders', None) != 0
                    or getattr(req, 'spec_token_ids', None) != []
                    or rid in single._partial_hit_reqs):
                reasons.append('UNSUPPORTED_REQUEST_LAYOUT:' + rid)
            return (rid, req.status.name, n, len(blocks), c, i, q,
                    rid in running_ids, rid in inflight_ids, identity, plain)

        # Read each physical owner once even when it belongs to both sets.
        rows = {rid: read(req) for rid, req in objects.items()}
        out['requests'] = [rows[rid] for rid in sorted(rows)]
        out['target'] = rows[target.request_id] if target.request_id in rows else read(target)
        if not reasons:
            out['status'] = 'KNOWN'
    except Exception as error:
        reasons.append('SNAPSHOT_EXCEPTION:' + type(error).__name__)
    out['return_host_perf_s'] = now()
    out['snapshot_s'] = out['return_host_perf_s'] - begin
    return out


def _decision(scheduler, manager, single, cs, mode, selective, now=time.perf_counter):
    data, original = _old_decision(scheduler, manager, single, cs, mode, selective, now)
    obs = data['progress_observation'] = dict(frozen_policy_sha256=FROZEN_SHA,
        scope='Only the first selected target: each own preallocate boundary through its first native continuation, at most 17 snapshots. No lookup/allocate/touch/CUDA; no growth or fit decision.',
        known_semantics='KNOWN means observed supported raw fields, not capacity fit or service guarantee; cohort finished does not imply either.',
        snapshots=[], snapshot_count=0, snapshot_s=0.0, release_event_indices=[],
        outcome='NO_SELECTION', runtime_qualification='Inherited pinned start_gate qualification plus per-snapshot owner/count/layout checks')
    target, stopped, step = None, False, 0

    def callback(current, phase, budget, preempted, queue=None, request=None, local=None,
                 new_tokens=None, external=None, asynchronous=None, local_tokens=None,
                 lookahead=None, encoder_tokens=None, new_blocks=None, reserved=None,
                 scheduled=None):
        nonlocal target, stopped, step
        if phase == 'entry':
            step += 1
        start = len(data['events'])
        result = original(current, phase, budget, preempted, queue, request, local,
                          new_tokens, external, asynchronous, local_tokens,
                          lookahead, encoder_tokens, new_blocks, reserved)
        for index in range(start, len(data['events'])):
            event = data['events'][index]
            if event['kind'] == 'legal_start_gate_opportunity':
                target = request
                obs['outcome'] = 'SELECTED_AWAITING_NATIVE_ATTEMPT_BOUNDARY'
            elif event['kind'] == 'release':
                obs['release_event_indices'].append(index)
        if phase == 'decision' and target is not None and request is target and not stopped:
            row = _snapshot(current, manager, single, target, scheduled, reserved, now)
            row.update(step=step,
                original_break_return=result, native_continuation_permitted=not result,
                original_event_indices=list(range(start, len(data['events']))),
                allocation_has_not_executed=True,
                native_arguments=dict(num_new_tokens=new_tokens, external_tokens=external,
                    load_kv_async=asynchronous, local_tokens=local_tokens,
                    lookahead_tokens=lookahead, encoder_tokens=encoder_tokens))
            obs['snapshots'].append(row)
            obs['snapshot_count'] += 1
            obs['snapshot_s'] += row['snapshot_s']
            if not result:
                stopped = True
                obs['outcome'] = 'NATIVE_ATTEMPT_BOUNDARY_OBSERVED'
        if phase == 'uninstall' and target is not None and not stopped:
            obs['outcome'] = 'NO_NATIVE_ATTEMPT_BOUNDARY_OBSERVED'
        return result
    return data, callback


# These overrides affect only this privately loaded module, never frozen source
# or another user's installed policy. Its install/attach/uninstall stay original.
G._instrument, G._decision = _instrument, _decision


def install(scheduler, mode='native', *, selective):
    return G.install(scheduler, mode, selective=selective)
