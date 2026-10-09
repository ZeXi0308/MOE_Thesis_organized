#!/usr/bin/env python3
"""Frozen fixed1024/compact/GC child; only age8 versus stall8 selection differs."""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
PARENT_SHA='e80dbfa7524f0f12bda24a0768bdf4270d619127b15a0e8d7f772db4c3a7a576'
POLICY_SHA='fd9c358d7aae9dfb475e5e06930d66c556ab2fed2cff74e82be1a481e5349b2f'
path=BASE/'recovery_repeat_unique/run_cell.py'
if hashlib.sha256(path.read_bytes()).hexdigest()!=PARENT_SHA: raise RuntimeError('Frozen common child changed')
spec=importlib.util.spec_from_file_location('service_age_common_child',path)
PARENT=importlib.util.module_from_spec(spec);spec.loader.exec_module(PARENT)


def adapt_source(text):
    text=PARENT.adapt_source(text)
    for old,new in (('repeat_unique','service_age'),('REPEAT_UNIQUE','SERVICE_AGE'),
                    ('repeat-unique','service-age')):
        if old not in text: raise RuntimeError('Service-age child boundary changed: '+old)
        text=text.replace(old,new)
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1: raise RuntimeError('Service-age capture boundary changed: '+old[:80])
        text=text.replace(old,new)
    replace(PARENT.POLICY_SHA,POLICY_SHA)
    replace(hashlib.sha256(Path(PARENT.__file__).read_bytes()).hexdigest(),
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    replace('        from compact_capture import get_capture, annotate',
        '        import compact_capture\n        from compact_capture import annotate\n'
        '        from service_age import make_capture')
    replace("        measure_episode = get_capture(os.environ['B_OUTPUT_EVENT_STORAGE'])",
        '        last_receipts = {}\n        measure_episode = make_capture(compact_capture, last_receipts)')
    replace("install_recovery_service_age(scheduler, os.environ['B_RECOVERY_SERVICE_AGE'])",
        "install_recovery_service_age(scheduler, os.environ['B_RECOVERY_SERVICE_AGE'], last_receipts=last_receipts)")
    anchor="        config['B_recovery_service_age'] = os.environ['B_RECOVERY_SERVICE_AGE']"
    replace(anchor,anchor+"\n        config['service_age_capture_sha256'] = measure_episode.service_age_capture_sha256")
    anchor="                annotate(raw, os.environ['B_OUTPUT_EVENT_STORAGE'])"
    replace(anchor,anchor+"\n                raw['output_event_storage']['base_compact_compiled_source_sha256'] = raw['output_event_storage']['compiled_source_sha256']"
        +"\n                raw['output_event_storage']['compiled_source_sha256'] = measure_episode.service_age_capture_sha256"
        +"\n                raw['service_age_capture'] = dict(compiled_source_sha256=measure_episode.service_age_capture_sha256, receipt_id_count=len(last_receipts), clock='origin + received after sync engine.step and valid positive prefix delta; no added clock')")
    return text


def main(capture_source=None):
    if POLICY_SHA is None: raise RuntimeError('Service-age policy is not frozen')
    source=inspect.getsource(PARENT.main)
    for old,new in (('B_RECOVERY_REPEAT_UNIQUE','B_RECOVERY_SERVICE_AGE'),
                    ('repeat8','age8'),('unique8','stall8'),('repeat_unique.py','service_age.py')):
        if old not in source: raise RuntimeError('Service-age main boundary changed: '+old)
        source=source.replace(old,new)
    namespace=dict(PARENT.__dict__,ROOT=ROOT,POLICY_SHA=POLICY_SHA,adapt_source=adapt_source)
    exec(compile(source,str(__file__)+'[frozen-child-main]','exec'),namespace)
    return namespace['main'](capture_source)


if __name__=='__main__':raise SystemExit(main())
