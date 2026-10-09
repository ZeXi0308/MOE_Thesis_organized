"""Reuse seeded private caches and the same bounded whole-group lock/handoff."""
import os
import run_native_max_release_group as group
import run_native_host_group_seeded as seeded


def main():
    seeded.group = group
    seeded.handoff.group = group
    print(f'MAX_RELEASE_ENTRY controller_pid={os.getpid()} '
          f'entry_sha256={group.base.digest(__file__)}', flush=True)
    return seeded.main()


if __name__ == '__main__':
    raise SystemExit(main())
