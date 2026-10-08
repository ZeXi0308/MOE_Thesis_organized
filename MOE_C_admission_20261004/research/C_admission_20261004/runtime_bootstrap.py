"""Process-local repair for orphan files left by a Torch upgrade; no install edits.

The shared Torch2.11 RECORD excludes three old Inductor kernel modules, whose
automatic import duplicates template registrations. Only import recorded modules.
"""
import importlib
import importlib.metadata
from pathlib import Path


def apply():
    import torch._dynamo.utils as du
    original = du.import_submodule
    if getattr(original, '_moe_c_recorded_kernel_import', False):
        return list(original._moe_c_skipped_kernel_modules)

    dist = importlib.metadata.distribution('torch')
    if dist.files is None:
        raise RuntimeError('Torch package RECORD is required for the C import repair')
    recorded = {str(p) for p in dist.files}
    root = Path(dist.locate_file('torch'))
    skipped = tuple(sorted(
        p.name for p in (root / '_inductor/kernel').glob('*.py')
        if not p.name.startswith('_')
        and str(p.relative_to(root.parent)) not in recorded
    ))

    def recorded_import(mod):
        if mod.__name__ != 'torch._inductor.kernel':
            return original(mod)
        for path in sorted(Path(mod.__file__).parent.glob('*.py')):
            if not path.name.startswith('_') and str(path.relative_to(root.parent)) in recorded:
                importlib.import_module(f'{mod.__name__}.{path.stem}')
    recorded_import._moe_c_recorded_kernel_import = True
    recorded_import._moe_c_skipped_kernel_modules = skipped
    du.import_submodule = recorded_import
    return list(skipped)
