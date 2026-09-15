try:
    raw = capture_with_memory(engine, capture_episode, workload, config,
        allow_preemption=True,
        regime='steady', arrival_scale=1.0, run_id='measured', max_seconds=120)
    dump(out/'post-request-drain.json', drain_offload(engine))
finally:
    dump(out/'selective-store.json', uninstall_selective())
    dump(out/'offload-events.json', offload_data)
    uninstall()
    dump(out/'headroom-decisions.json', decisions)
if args.completion_policy == 'headroom' and raw['status'] == 'COMPLETE':
    if not any(d['held'] for d in decisions):
        raw.update(status='INVALID_NO_ACTION', error='headroom never held a request')
if args.completion_policy == 'rotate' and raw['status'] == 'COMPLETE':
    if not any(d.get('forced_preempted') for d in decisions):
        raw.update(status='INVALID_NO_ACTION', error='rotation never applied a forced preemption')
if raw['status']=='COMPLETE' and selective_data['applied_rotations']<2:
    raise RuntimeError('Repeated intervention did not execute at least twice')
raw['gpu_before'] = before
dump(out/'raw.json', raw)
