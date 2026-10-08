"""Small CPU arithmetic checks; no runtime, GPU or native policy installation."""
import copy
from progress_capacity import estimate


def row(rid, n, held, computed=0, inflight=0, scheduled=0):
    return dict(request=rid, history_tokens=n, held_gpu_blocks=held,
                computed_tokens=computed, num_in_flight_tokens=inflight,
                scheduled_tokens_this_step=scheduled)


base = dict(target=row('target', 49, 0),
    requests=[row('load', 65, 2, 32), row('prefill', 65, 3, 32, 16, 16), row('decode', 31, 2, 30, 0, 1)],
    running_ids=['prefill', 'decode'], inflight_ids=['load', 'prefill'],
    native_reserved_blocks=5, free_gpu_blocks=12, block_size=16, max_model_len=4096,
    growth_tokens=16, target_growth_tokens=0, qualified=True)
before = copy.deepcopy(base)
result = estimate(**base)
assert base == before  # No input mutation, allocation or state touch.
assert result['status'] == 'KNOWN' and result['capacity_fit']
assert (result['target_total_blocks'], result['inflight_remaining_blocks'],
        result['running_increment_blocks'], result['required_free_blocks']) == (4,5,2,11)
prefill = next(r for r in result['per_request'] if r['request'] == 'prefill')
assert (prefill['inflight_remaining_blocks'], prefill['running_interval_deficit_blocks'],
        prefill['reservation_overlap_blocks'], prefill['running_increment_blocks']) == (2,3,2,1)
assert estimate(**(base | dict(free_gpu_blocks=10)))['capacity_fit'] is False
zero = estimate(**(base | dict(growth_tokens=0)))
assert zero['required_free_blocks'] == 9 and zero['running_increment_blocks'] == 0
growing_target = estimate(**(base | dict(target_growth_tokens=16)))
assert growing_target['target_current_history_blocks'] == 4
assert growing_target['target_growth_increment_blocks'] == 1
assert growing_target['required_free_blocks'] == 12

# A whole block of extra positions with fully allocated histories, without a
# context cap, is exactly one extra block per running request: a simple baseline.
whole = base | dict(requests=[row('a',16,1,15,0,1),row('b',31,2,30,0,1),row('c',47,3,46,0,1)],
    running_ids=['a','b','c'], inflight_ids=[], native_reserved_blocks=0)
simple = estimate(**whole)
assert simple['running_increment_blocks'] == 3
assert all(r['running_increment_blocks'] == 1 for r in simple['per_request'])
cap = base | dict(target=row('target',49,0), requests=[row('cap',4095,256,4094,0,1)],
    running_ids=['cap'], inflight_ids=[], native_reserved_blocks=0)
assert estimate(**cap)['running_increment_blocks'] == 0

for change in (dict(qualified=False), dict(native_reserved_blocks=6),
               dict(inflight_ids=['load','prefill','load']), dict(growth_tokens=None),
               dict(target=row('load',49,0)), dict(target=row('target',49,1)),
               dict(requests=[row('load',65,2,32),row('prefill',65,3,32,33,16),row('decode',31,2,30,0,1)])):
    unknown = estimate(**(base | change))
    assert unknown['status'] == 'UNKNOWN' and unknown['capacity_fit'] is None
try:
    estimate(**{k:v for k,v in base.items() if k != 'growth_tokens'})
    raise AssertionError('growth_tokens must have no default')
except TypeError:
    pass
try:
    estimate(**{k:v for k,v in base.items() if k != 'target_growth_tokens'})
    raise AssertionError('target_growth_tokens must have no default')
except TypeError:
    pass
print('PASS: three disjoint capacity terms, physical-prefix/current-step/inflight dedup, target growth, bounds/UNKNOWN, and growth=B equivalence to one block per running request. CPU only; no policy installed.')
