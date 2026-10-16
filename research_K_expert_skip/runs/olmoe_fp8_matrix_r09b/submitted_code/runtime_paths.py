"""Locate the single-source JSON assets in a checkout or pip installation.

Setuptools maps configs/ into the data-only k_expert_skip_configs namespace
package. Both ordinary wheel installs and pip --target put that directory
beside this module; no installation-prefix or working-directory assumption
is needed. Source execution continues to read the original configs/ files.
"""
from pathlib import Path


def config_path(name: str) -> Path:
    """Return an existing bundled JSON asset as a filesystem path."""
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError("config name must be a single .json filename")
    module_dir = Path(__file__).resolve().parent
    installed = module_dir / "k_expert_skip_configs" / name
    if installed.is_file():
        return installed
    source = module_dir.parent / "configs" / name
    if (module_dir.name == "code" and (module_dir.parent / "pyproject.toml").is_file()
            and source.is_file()):
        return source
    raise FileNotFoundError(f"Bundled configuration {name!r} is missing; reinstall marginal-expert-skip")
