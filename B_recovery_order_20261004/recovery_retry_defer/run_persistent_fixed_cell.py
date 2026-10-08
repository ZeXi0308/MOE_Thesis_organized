#!/usr/bin/env python3
"""Keep fixed1024 unchanged while selecting the persistent retry cell."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = 'b5c92085b0401c7803c968d6902ebcc9d35464fa13a12c478a267fd7bf67d0a3'


def termination_source(text):
    path = ROOT/'run_fixed_cell.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen retry-defer fixed-output cell changed')
    spec = importlib.util.spec_from_file_location('persistent_retry_fixed_parent', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    text = parent.termination_source(text)
    old = "        config['recovery_lease_mode']=lease_mode"
    if text.count(old) != 1:
        raise RuntimeError('Persistent fixed-output provenance boundary changed')
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return text.replace(old, f"        config['recovery_retry_defer_persistent_fixed_cell_sha256']={digest!r}\n"+old)


def main():
    spec = importlib.util.spec_from_file_location('persistent_retry_cell', ROOT/'run_persistent_cell.py')
    cell = importlib.util.module_from_spec(spec); spec.loader.exec_module(cell)
    return cell.main(source_transform=termination_source)


if __name__ == '__main__':
    raise SystemExit(main())
