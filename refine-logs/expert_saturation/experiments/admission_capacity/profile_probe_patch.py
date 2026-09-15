"""Generate a diagnostic-only profiler copy of the frozen phase probe."""


def patch_profile_probe(source: str) -> str:
    def patch(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError("profile probe anchor changed: " + old[:80])
        source = source.replace(old, new, 1)

    patch("    def save():\n", '''    profiler = None
    result["profile"] = dict(status="NOT_STARTED", path=str(args.out.with_name("profile.json")),
                             diagnostic_only=True, timings_include_profiler_overhead=True)
    def finish_profile(partial=False):
        nonlocal profiler
        if profiler is None:
            return
        active, profiler = profiler, None
        try:
            active.stop()
            active.export_chrome_trace(result["profile"]["path"])
            result["profile"]["status"] = "PARTIAL_EXPORTED" if partial else "EXPORTED"
        except Exception as error:
            result["profile"].update(status="FAILED", error=f"{type(error).__name__}: {error}")
            if not partial:
                raise
    def save():
''')
    patch("        origin, wall_origin = time.perf_counter(), time.time()\n", '''        require(args.policy == "static8", "Profiler campaign is static8 diagnostic only")
        result["profile"]["status"] = "STARTING"
        if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
            result["profile"]["status"] = "FAILED"
            raise RuntimeError("CUDA profiler activity unavailable; diagnostic cannot run")
        if Path(result["profile"]["path"]).exists():
            raise FileExistsError(result["profile"]["path"])
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False, with_stack=False)
        profiler.start()
        result["profile"]["status"] = "RECORDING"
        origin, wall_origin = time.perf_counter(), time.time()
''')
    patch("                outputs = engine.step()\n", '''                with torch.profiler.record_function(f"ENGINE_CALL/{call['index']}"):
                    outputs = engine.step()
''')
    patch('        result["wall_s"] = now()\n', '        result["wall_s"] = now()\n        finish_profile()\n')
    patch("    except (Exception, KeyboardInterrupt) as exc:\n",
          "    except (Exception, KeyboardInterrupt) as exc:\n        finish_profile(partial=True)\n")
    patch("Host receipt; includes inline observation and action snapshot costs",
          "DIAGNOSTIC: host receipt includes profiler, inline observation and action snapshot costs")
    compile(source, "profiled_phase_injection_probe.py", "exec")
    return source
