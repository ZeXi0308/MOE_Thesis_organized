from __future__ import annotations

from pathlib import Path


BASE_OUT = Path("outputs")  # prefer idea-local outputs/; override via CLI


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "--").replace(":", "_")


def default_run_dir(model_name: str) -> Path:
    return BASE_OUT / "runs" / model_slug(model_name)


def resolve_output_dir(model_name: str, output_dir: str | None) -> Path:
    return Path(output_dir) if output_dir else default_run_dir(model_name)

