#!/usr/bin/env python3
"""Bind only the measured host lock identity over the frozen ACK controller."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = 'e99c8c2f0a9b75b42b81d4049f81127f2cceb0aafde3c4bf241ecc0a3ece3fc1'
LOCK_PATH = '/root/autodl-tmp/moe-research-gpu.lock'
LOCK_IDENTITY = (2304, 4312099778)


def load_parent():
    path = ROOT/'run_ack_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen ACK-yield controller changed')
    spec = importlib.util.spec_from_file_location('ack_host_group_parent', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    load_start_parent = parent.load_parent

    def start_parent():
        start = load_start_parent()
        load_resource_parent = start.load_parent

        def resource_parent():
            resource = load_resource_parent()
            resource.LOCK_PATH = LOCK_PATH
            resource.LOCK_IDENTITY = LOCK_IDENTITY
            return resource

        start.load_parent = resource_parent
        return start

    parent.load_parent = start_parent
    parent.__file__ = str(__file__)
    return parent


def adapted_source():
    return load_parent().adapted_source()


def main():
    return load_parent().main()


if __name__ == '__main__':
    raise SystemExit(main())
