#!/usr/bin/env python3
"""Native/stall8 using unchanged fixed1024, client receipt, compact and GC capture."""
import hashlib
import importlib.util
import inspect
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
PARENT_SHA='cd8e577b474a9ef83127576fdfb21dddcae408c1be6d92149edf50113a3bf073'
POLICY_SHA='2dfed438f359c821c30d789f2f6c8aa6dbe9b0b091be54b0d1becfb15030e031'
path=BASE/'recovery_service_age/run_cell.py'
if hashlib.sha256(path.read_bytes()).hexdigest()!=PARENT_SHA:
    raise RuntimeError('Frozen service-age child changed')
spec=importlib.util.spec_from_file_location('native_service_age_child',path)
FROZEN=importlib.util.module_from_spec(spec);spec.loader.exec_module(FROZEN)


def adapt_source(text):
    text=FROZEN.adapt_source(text)
    changes={
        'from service_age import':'from native_service_age import',
        'B_RECOVERY_SERVICE_AGE':'B_RECOVERY_SERVICE_AGE_NATIVE',
        "config['B_recovery_service_age']":"config['B_recovery_service_age_native']",
        "config['recovery_service_age_policy_sha256']":"config['recovery_service_age_native_policy_sha256']",
        "config['recovery_service_age_runner_sha256']":"config['recovery_service_age_native_runner_sha256']",
        "out/'recovery-service-age.json'":"out/'recovery-service-age-native.json'",
        FROZEN.POLICY_SHA:POLICY_SHA,
        hashlib.sha256(Path(FROZEN.__file__).read_bytes()).hexdigest():hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for old,new in changes.items():
        expected=2 if old in ('from service_age import','B_RECOVERY_SERVICE_AGE') else 1
        if text.count(old)!=expected:raise RuntimeError('Native service-age child boundary changed: '+old)
        text=text.replace(old,new)
    return text


def main(capture_source=None):
    if POLICY_SHA is None:raise RuntimeError('Native service-age policy is not frozen')
    # Reuse the same original fixed/GC/compact child entry; only effective mode,
    # policy filename, and final source adapter differ from the frozen child.
    source=inspect.getsource(FROZEN.PARENT.main)
    for old,new in (('B_RECOVERY_REPEAT_UNIQUE','B_RECOVERY_SERVICE_AGE_NATIVE'),
                    ('repeat8','native'),('unique8','stall8'),('repeat_unique.py','native_service_age.py')):
        if old not in source:raise RuntimeError('Native service-age main boundary changed: '+old)
        source=source.replace(old,new)
    namespace=dict(FROZEN.PARENT.__dict__,ROOT=ROOT,POLICY_SHA=POLICY_SHA,adapt_source=adapt_source)
    exec(compile(source,str(__file__)+'[frozen-common-child-main]','exec'),namespace)
    return namespace['main'](capture_source)


if __name__=='__main__':raise SystemExit(main())
