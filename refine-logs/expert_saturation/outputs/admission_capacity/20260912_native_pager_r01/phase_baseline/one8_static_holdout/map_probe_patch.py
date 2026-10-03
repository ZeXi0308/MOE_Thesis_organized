"""Attach map-only instrumentation outside the frozen probe's timing boundary."""


def patch_map_probe(source):
    def patch(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError("map probe source anchor changed: " + old[:80])
        source = source.replace(old, new, 1)

    patch('        engine = llm.llm_engine\n', '''        engine = llm.llm_engine
        import batched_expert_map as map_patch
        map_patch.prepare()
''')
    patch('        origin, wall_origin = time.perf_counter(), time.time()\n', '''        result["map_initial"] = map_patch.stats(reset=True)
        origin, wall_origin = time.perf_counter(), time.time()
''')
    patch('        result["wall_s"] = now()\n', '''        result["wall_s"] = now()
        result["map_measurement"] = map_patch.stats()
        result["map_validation"] = map_patch.validate()
''')
    compile(source, "map_injection_probe.py", "exec")
    return source
