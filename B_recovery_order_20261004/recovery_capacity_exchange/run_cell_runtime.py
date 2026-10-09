#!/usr/bin/env python3
"""Frozen r01 child with exact dynamic-source guarded exchange import substituted."""
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
PARENT_SHA='69578cfaacd826ddca0995fabff15b59d6fc1b1f00d0716de0483d4874d3f73f'
POLICY_SHA='fa1decbde3afbcbdf51d479b234cae832aeb25ad3084939d144d4267d2ac7ac3'
path=ROOT/'run_cell.py'
if hashlib.sha256(path.read_bytes()).hexdigest()!=PARENT_SHA:
    raise RuntimeError('Frozen r01 exchange child changed')
spec=importlib.util.spec_from_file_location('guarded_exchange_child_parent',path)
PARENT=importlib.util.module_from_spec(spec);spec.loader.exec_module(PARENT)


def adapt_source(text):
    text=PARENT.adapt_source(text)
    for old,new,count in (
        ('from exchange_once import','from exchange_guarded_runtime import',2),
        (PARENT.POLICY_SHA,POLICY_SHA,1),
        (PARENT_SHA,hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),1)):
        if text.count(old)!=count:raise RuntimeError('Guarded child boundary changed: '+old)
        text=text.replace(old,new)
    return text


def main(capture_source=None):
    source=inspect.getsource(PARENT.main)
    if source.count("'exchange_once.py'")!=1:raise RuntimeError('Expected one policy entry pin')
    source=source.replace("'exchange_once.py'","'exchange_guarded_runtime.py'")
    namespace=dict(PARENT.__dict__,POLICY_SHA=POLICY_SHA,adapt_source=adapt_source,__file__=str(__file__))
    exec(compile(source,str(__file__)+'[frozen-r01-main]','exec'),namespace)
    return namespace['main'](capture_source)


def self_check():
    class Captured(Exception):pass
    def capture(text):
        compile(text,'<complete-guarded-child>','exec')
        assert text.count('from exchange_guarded_runtime import')==2
        assert 'from exchange_once import' not in text
        assert text.count("install_recovery_service_age(scheduler, os.environ['B_RECOVERY_CAPACITY_EXCHANGE'], last_receipts=last_receipts, selective=selective)")==1
        assert POLICY_SHA in text and PARENT.POLICY_SHA not in text
        assert "measure_episode = make_capture(compact_capture, last_receipts)" in text
        assert text.count('install_gc_probe()')==1
        assert 'ignore_eos=True, min_tokens=0,' in text
        assert 'output_tokens=16, output_tokens_by_request={}' in text
        raise Captured
    old=dict(os.environ)
    try:
        for mode in ('stall8','exchange_once'):
            os.environ['B_RECOVERY_CAPACITY_EXCHANGE']=mode
            try:main(capture)
            except Captured:pass
            else:raise AssertionError('Missing stop before native execution')
    finally:os.environ.clear();os.environ.update(old)
    print('PASS: full fixed1024/compact/GC child source; only guarded import/policy/runner provenance differ; both modes. CPU only.')


if __name__=='__main__':
    if sys.argv[1:]==['--self-check']:self_check()
    else:raise SystemExit(main())
