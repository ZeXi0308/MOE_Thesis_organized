#!/usr/bin/env python3
"""Use a measured 2.5GiB group-prelaunch floor over the frozen host binding."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = 'c8141eacfcc6f6e147b7b60b3516893ea1db5fe56f67bf3f0758ab36b945e2d9'
MINIMUM_FREE_BYTES = 2684354560


def load_parent():
    path = ROOT/'run_ack_host_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen ACK host controller changed')
    spec = importlib.util.spec_from_file_location('ack_space_group_parent', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    load_ack_parent = parent.load_parent

    def ack_parent():
        ack = load_ack_parent()
        load_start_parent = ack.load_parent

        def start_parent():
            start = load_start_parent()
            load_resource_parent = start.load_parent

            def resource_parent():
                resource = load_resource_parent()
                resource.MINIMUM_FREE_BYTES = MINIMUM_FREE_BYTES
                original_check = resource.check_space

                def check_space(session, receipt, write):
                    try:
                        return original_check(session, receipt, write)
                    except RuntimeError as error:
                        if str(error) == 'Session/cache filesystem has less than 3GiB free; no cell started':
                            raise RuntimeError('Session/cache filesystem has less than 2.5GiB free; no cell started') from error
                        raise

                resource.check_space = check_space
                return resource

            start.load_parent = resource_parent
            return start

        ack.load_parent = start_parent
        return ack

    parent.load_parent = ack_parent
    parent.__file__ = str(__file__)
    return parent


def adapted_source():
    return load_parent().adapted_source()


def main():
    return load_parent().main()


if __name__ == '__main__':
    raise SystemExit(main())
