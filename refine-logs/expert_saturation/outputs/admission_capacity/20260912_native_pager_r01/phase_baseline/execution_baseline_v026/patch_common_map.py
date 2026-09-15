"""Apply the already qualified map-copy optimization equally to both executors."""


def patch_common_map(source):
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError("common map source anchor changed: " + old[:70])
        source = source.replace(old, new, 1)

    replace('        runtime = pager.install(args.expert_cap, out / "pager")', '''        import batched_expert_map as common_map
        common_map.set_mode("batched")
        dump(out / "map_selftest.json", common_map.gpu_selftest())
        runtime = pager.install(args.expert_cap, out / "pager")''')
    replace('            pager.begin_measurement(phase)', '''            for entry in runtime.layers.values():
                common_map._entry(entry["state"])
            common_map.stats(reset=True)
            pager.begin_measurement(phase)''')
    replace('            dump(episode_out / "raw.json", raw)', '''            dump(episode_out / "raw.json", raw)
            dump(episode_out / "map_optimization.json", dict(
                statistics=common_map.stats(), validation=common_map.validate(),
                scope="Same batched full resident map update in token and expert arms; expert kernel still receives a separate masked group map. Validation runs after capture wall."))''')
    return source
