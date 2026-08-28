from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "jamesdborin/tiny-mixtral"


def load_tokenizer(
    model_name: str = DEFAULT_MODEL,
    local_files_only: bool = False,
    revision: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only, revision=revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def resolve_dtype(dtype_name: str):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "auto":
        return "auto"
    raise ValueError(f"unknown dtype: {dtype_name}")


def resolve_device() -> str:
    """Backward compatible: Mac runs had no explicit device move (CPU-only,
    since torch.cuda.is_available() is False there). On a CUDA machine this
    moves the model to GPU; behaviour on non-CUDA machines is unchanged."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(
    model_name: str = DEFAULT_MODEL,
    dtype_name: str = "float32",
    local_files_only: bool = False,
    revision: str | None = None,
):
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=resolve_dtype(dtype_name),
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
        revision=revision,
    )
    model.eval()
    device = resolve_device()
    if device != "cpu":
        model = model.to(device)
    elapsed = time.time() - start
    return model, elapsed
