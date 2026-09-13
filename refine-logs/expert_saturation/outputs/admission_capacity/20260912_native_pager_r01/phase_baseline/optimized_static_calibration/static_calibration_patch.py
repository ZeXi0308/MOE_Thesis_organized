"""Derive static8/16/32 calibration copies from the frozen phase baseline."""


def _replace(source, old, new, count=1):
    if source.count(old) != count:
        raise RuntimeError("static calibration source anchor changed: " + old[:80])
    return source.replace(old, new, count)


def patch_static_probe(source: str) -> str:
    for old, new in (
        ('choices=(8, 32), required=True', 'choices=(8, 16, 32), required=True'),
        ('choices=("static8", "phase8"), required=True',
         'choices=("static8", "static16", "static32"), required=True'),
        ('require(args.chunk == 8 and args.new_prompt_length == 128 and not args.no_new and not args.trace,',
         'require(args.policy == f"static{args.chunk}" and args.new_prompt_length == 128 and not args.no_new and not args.trace,'),
        ('Phase baseline requires chunk8/long128/injection/trace=false',
         'Static calibration requires matching static8/16/32 and chunk, long128/injection/trace=false'),
    ):
        source = _replace(source, old, new)
    compile(source, "static_calibration_probe.py", "exec")
    return source


def patch_static_policy(source: str) -> str:
    source = _replace(source, 'mode not in ("phase8", "static8")',
                      'mode not in ("static8", "static16", "static32")', count=2)
    source = _replace(source, 'mode must be phase8 or static8',
                      'mode must be static8, static16 or static32', count=2)
    source = _replace(source, '(8 if mode == "static8" or ready else 0)',
                      'int(mode[len("static"):])')
    source = _replace(source, 'Use only current running progress; 0 removes the per-request cap.',
                      'Keep threshold32 before the action; then use the validated static cap.')
    compile(source, "static_calibration_policy.py", "exec")
    return source
