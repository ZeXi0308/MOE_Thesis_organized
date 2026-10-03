"""Optional source-fenced hooks; caller separately installs Layer47Replay on runtime."""
import hashlib


def edit(source, sha, old, new):
    if hashlib.sha256(source.encode()).hexdigest() != sha or source.count(old) != 1:
        raise ValueError("qualification replay source/anchor mismatch")
    return source.replace(old, new)


def patch_expert_groups(source):
    old = '                global_num_experts=state.num_experts, expert_map=group_map)\n            host_start = time.perf_counter()'
    new = '''                global_num_experts=state.num_experts, expert_map=group_map)
            if validate and getattr(self, "numerical_localization", None) is not None:
                self.numerical_localization.retain(record, group_map, y)
            host_start = time.perf_counter()'''
    return edit(source, "feb13bfe023fb36f6b171e74ce0882ee49d3f1df5d50f097759c5e5aa0f88410", old, new)


def patch_adapter(source):
    old = '        del w13, w2, reference'
    new = '''        if getattr(self, "numerical_localization", None) is not None:
            self.numerical_localization.finish(self, method, layer, fixed, actual, reference, w13, w2, record)
        del w13, w2, reference'''
    return edit(source, "b6c25e2656678636a596403fba7d9318465a4bee5b0eaf9c7b457dadcbafdd93", old, new)
