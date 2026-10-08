#!/usr/bin/env python3
"""Use the persistent active-head guard over the frozen retry-defer cell."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = '2ad4e5c8a6608068725b03b8b636716b2b34afe5a38b31b2143af67babf8cbfb'
POLICY_SHA = '4eae969ca391e01d8ab44c516115f9fdc702e12537a01240576c0484b98579e8'


def load_parent():
    path = ROOT/'run_cell.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen retry-defer cell changed')
    policy = ROOT/'retry_defer_persistent.py'
    if POLICY_SHA is None or not policy.exists() or hashlib.sha256(policy.read_bytes()).hexdigest() != POLICY_SHA:
        raise RuntimeError('Persistent policy is not frozen or differs from its pin')
    spec = importlib.util.spec_from_file_location('persistent_retry_native_cell', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    return parent


def adapt_source(text):
    changes = {
        '        from retry_defer import install as install_retry_defer':
            '        from retry_defer_persistent import install as install_retry_defer',
        repr(str(ROOT/'retry_defer.py')): repr(str(ROOT/'retry_defer_persistent.py')),
        "        config['B_recovery_retry_defer'] = os.environ['B_RECOVERY_RETRY_DEFER']":
            "        config['B_recovery_retry_defer'] = os.environ['B_RECOVERY_RETRY_DEFER']\n"
            "        config['recovery_retry_defer_revision'] = 'persistent_active_head_passthrough'\n"
            f"        config['recovery_retry_defer_persistent_runner_sha256'] = {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!r}",
    }
    for old, new in changes.items():
        if text.count(old) != 1:
            raise RuntimeError('Persistent retry cell boundary changed: '+old[:90])
        text = text.replace(old, new)
    return text


def main(source_transform=None):
    parent = load_parent(); original = parent.adapt_source
    parent.adapt_source = lambda text: adapt_source(original(text))
    return parent.main(source_transform=source_transform)


if __name__ == '__main__':
    raise SystemExit(main())
