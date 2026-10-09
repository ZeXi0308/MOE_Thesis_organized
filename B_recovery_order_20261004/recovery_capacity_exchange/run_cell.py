#!/usr/bin/env python3
"""One funded capacity exchange over the frozen stall8 fixed-workload child."""
import hashlib
import importlib.util
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT_SHA = 'cd8e577b474a9ef83127576fdfb21dddcae408c1be6d92149edf50113a3bf073'
POLICY_SHA = 'ce54ed804c757d936c945741b1f19b8e8906053836a03630d978e7e4d771ca97'
path = BASE/'recovery_service_age/run_cell.py'
if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
    raise RuntimeError('Frozen service-age child changed')
spec = importlib.util.spec_from_file_location('capacity_exchange_child_parent', path)
FROZEN = importlib.util.module_from_spec(spec); spec.loader.exec_module(FROZEN)


def adapt_source(text):
    text = FROZEN.adapt_source(text)
    changes = {
        'from service_age import': 'from exchange_once import',
        'B_RECOVERY_SERVICE_AGE': 'B_RECOVERY_CAPACITY_EXCHANGE',
        "config['B_recovery_service_age']": "config['B_recovery_capacity_exchange']",
        "config['recovery_service_age_policy_sha256']": "config['recovery_capacity_exchange_policy_sha256']",
        "config['recovery_service_age_runner_sha256']": "config['recovery_capacity_exchange_runner_sha256']",
        "out/'recovery-service-age.json'": "out/'recovery-capacity-exchange.json'",
        FROZEN.POLICY_SHA: POLICY_SHA,
        hashlib.sha256(Path(FROZEN.__file__).read_bytes()).hexdigest(): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for old, new in changes.items():
        expected = 2 if old in ('from service_age import', 'B_RECOVERY_SERVICE_AGE') else 1
        if text.count(old) != expected:
            raise RuntimeError('Capacity-exchange child boundary changed: '+old)
        text = text.replace(old, new)
    old = "install_recovery_service_age(scheduler, os.environ['B_RECOVERY_CAPACITY_EXCHANGE'], last_receipts=last_receipts)"
    if text.count(old) != 1:
        raise RuntimeError('Expected one new policy install with shared receipts')
    text = text.replace(old, old[:-1]+', selective=selective)')
    return text


def main(capture_source=None):
    if POLICY_SHA is None:
        raise RuntimeError('Capacity-exchange policy is not frozen; no GPU launch')
    source = inspect.getsource(FROZEN.PARENT.main)
    for old, new in (('B_RECOVERY_REPEAT_UNIQUE', 'B_RECOVERY_CAPACITY_EXCHANGE'),
                     ('repeat8', 'stall8'), ('unique8', 'exchange_once'),
                     ('repeat_unique.py', 'exchange_once.py')):
        if old not in source:
            raise RuntimeError('Capacity-exchange child entry changed: '+old)
        source = source.replace(old, new)
    namespace = dict(FROZEN.PARENT.__dict__, ROOT=ROOT, POLICY_SHA=POLICY_SHA,
                     adapt_source=adapt_source)
    exec(compile(source, str(__file__)+'[frozen-common-child-main]', 'exec'), namespace)
    return namespace['main'](capture_source)


if __name__ == '__main__':
    raise SystemExit(main())
