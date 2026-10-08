"""One frozen ABBA diagnostic with identical private Triton cache seeds."""
import os
from pathlib import Path

import run_native_group_seeded_base as seeded_base
import run_native_host_group as group
import run_native_host_group_handoff as handoff
import triton_seed


def main():
    group.base = seeded_base
    original_configure = group.configure

    def configure(plan):
        seeded_base.require(plan['runtime_cache_initial'] == 'TRITON_SEEDED_PRIVATE_OTHER_CACHES_EMPTY',
                            'Must explicitly record seeded initial cache state')
        seed = Path(plan['triton_cache_seed']['path'])
        expected = plan['triton_cache_seed']['normalized_sha256']
        seeded_base.require(seed.parent == group.OUTPUT_ROOT and seed.is_dir(),
                            'Seed must be in A output root')
        state = triton_seed.inspect_seed(seed)
        seeded_base.require(state['normalized_sha256'] == expected, 'Triton seed differs')
        original_configure(plan)
        original_env = seeded_base.private_env

        def env(p, cache, arm, memory_file):
            result = original_env(p, cache, arm, memory_file)
            if arm != 'ordinary':
                record = triton_seed.install(seed, cache / 'triton', expected)
                record.update(source=str(seed), rule=result['A_NATIVE_VICTIM_RULE'],
                              cache_initial=p['runtime_cache_initial'])
                seeded_base.write_json(cache.parent / 'triton-cache-seed.json', record, new=True)
            return result

        seeded_base.private_env = env

    group.configure = configure
    print(f'NATIVE_HOST_SEEDED_ENTRY controller_pid={os.getpid()} '
          f'entry_sha256={seeded_base.digest(__file__)}', flush=True)
    return handoff.main()


if __name__ == '__main__':
    raise SystemExit(main())
