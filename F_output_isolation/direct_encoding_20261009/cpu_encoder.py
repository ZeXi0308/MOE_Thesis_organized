"""Array-to-JSON CPU baseline shared token table and ctypes wrapper.

Only ordinary decoded-token top-k logprobs are handled. Token-ID placeholders,
prompt logprobs and other API variants must remain on the native path. Startup
token conversion uses vLLM's actual convert_ids_list_to_tokens. Any token whose
converted text ends with U+FFFD makes its whole row fall back to native UTF-8
context correction. No timer, new batching window or request ordering is used.

TokenTable ABI (also consumed by gpu_encoder.py):
  blob: bytes containing pre-encoded JSON pieces;
  offsets: contiguous uint64[V,2], lengths: contiguous uint32[V,2];
  part 0: {"token":<JSON string>,"logprob":
  part 1: ,"bytes":[<UTF-8 byte values>]       (no closing object brace)
  unsafe: contiguous uint8[V], 1 for context-dependent/unencodable tokens.

encode() takes contiguous int64[R,K+1] IDs and float32[R,K+1] logprobs. Column 0
is sampled; remaining columns are alternatives. Dict first-insertion order and
last-value overwrite semantics are preserved before selecting the first K top
entries. Returned offsets partition packed bytes into one content object per
row. Status 0=encoded, 1=native context fallback, 2=invalid ID, 3=NaN/+infinity.
Negative infinity is clamped to -9999 exactly as in native API construction.

Each encode uses ONE ctypes CDLL call, which releases the GIL. It then performs
one packed-buffer copy plus two small metadata copies. All copies belong in
end-to-end measurements. A per-encoder lock protects native reusable buffers.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
from typing import Callable, Iterable

import numpy as np


@dataclass
class TokenTable:
    blob: bytes
    offsets: np.ndarray
    lengths: np.ndarray
    unsafe: np.ndarray
    build_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.blob, bytes):
            raise TypeError("blob must be bytes")
        self.offsets = np.ascontiguousarray(self.offsets, dtype=np.uint64)
        self.lengths = np.ascontiguousarray(self.lengths, dtype=np.uint32)
        self.unsafe = np.ascontiguousarray(self.unsafe, dtype=np.uint8)
        if self.offsets.ndim != 2 or self.offsets.shape[1] != 2:
            raise ValueError("offsets must have shape [vocabulary,2]")
        if self.lengths.shape != self.offsets.shape:
            raise ValueError("lengths and offsets shape mismatch")
        if self.unsafe.shape != (len(self.offsets),):
            raise ValueError("unsafe must have shape [vocabulary]")
        # Subtraction avoids uint64 addition wrapping on malformed spans.
        if np.any(self.offsets > len(self.blob)) or np.any(
            self.lengths > len(self.blob) - self.offsets
        ):
            raise ValueError("token table span out of bounds")

    @property
    def vocab_size(self) -> int:
        return len(self.unsafe)

    @property
    def nbytes(self) -> int:
        return len(self.blob) + self.offsets.nbytes + self.lengths.nbytes + self.unsafe.nbytes

    def save(self, filename: str | Path) -> None:
        """Persist the SAME precomputed table for CPU and GPU initialization."""
        np.savez(filename, blob=np.frombuffer(self.blob, dtype=np.uint8),
                 offsets=self.offsets, lengths=self.lengths, unsafe=self.unsafe,
                 metadata=np.array(json.dumps({**self.metadata,
                                               "build_seconds": self.build_seconds})))

    @classmethod
    def load(cls, filename: str | Path) -> "TokenTable":
        with np.load(filename, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            build_seconds = meta.pop("build_seconds", 0.0)
            return cls(data["blob"].tobytes(), data["offsets"], data["lengths"],
                       data["unsafe"], build_seconds, meta)


def token_table_from_strings(decoded: Iterable[str], *, metadata=None) -> TokenTable:
    """Build JSON pieces, useful for ABI tests; production uses build_token_table."""
    started = time.perf_counter()
    pieces: list[bytes] = []
    offsets: list[tuple[int, int]] = []
    lengths: list[tuple[int, int]] = []
    unsafe: list[int] = []
    at = 0
    for token in decoded:
        bad = not isinstance(token, str) or token.endswith("\ufffd")
        try:
            prefix = b'{"token":' + json.dumps(token, ensure_ascii=False).encode("utf-8") + b',"logprob":'
            suffix = b',"bytes":[' + b",".join(
                str(x).encode("ascii") for x in token.encode("utf-8", errors="replace")
            ) + b"]"
        except (UnicodeEncodeError, AttributeError, TypeError):
            # Native handles unusual decoder outputs; never fabricate text.
            bad, prefix, suffix = True, b"", b""
        if len(prefix) > 2**32 - 1 or len(suffix) > 2**32 - 1:
            raise ValueError("token JSON piece exceeds uint32")
        offsets.append((at, at + len(prefix)))
        lengths.append((len(prefix), len(suffix)))
        pieces.extend((prefix, suffix))
        at += len(prefix) + len(suffix)
        unsafe.append(int(bad))
    if not offsets:
        raise ValueError("token table must be nonempty")
    table = TokenTable(b"".join(pieces), np.array(offsets, dtype=np.uint64),
                       np.array(lengths, dtype=np.uint32), np.array(unsafe, dtype=np.uint8),
                       metadata=dict(metadata or {}))
    table.build_seconds = time.perf_counter() - started
    table.metadata.update({"abi": 1, "vocab_size": table.vocab_size,
                           "unsafe_tokens": int(np.count_nonzero(table.unsafe)),
                           "table_nbytes": table.nbytes,
                           "blob_sha256": hashlib.sha256(table.blob).hexdigest()})
    return table


def build_token_table(tokenizer, converter: Callable | None = None,
                      vocab_size: int | None = None) -> TokenTable:
    started = time.perf_counter()
    if converter is None:
        from vllm.tokenizers.detokenizer_utils import convert_ids_list_to_tokens
        converter = convert_ids_list_to_tokens
    if vocab_size is None:
        # len(tokenizer) includes added tokens; get_vocab covers noncontiguous
        # tokenizer IDs. Extra model-only padded vocabulary IDs remain invalid.
        ids = tokenizer.get_vocab().values()
        vocab_size = max(int(x) for x in ids) + 1
    decoded = converter(tokenizer, list(range(vocab_size)))
    if len(decoded) != vocab_size:
        raise ValueError("native tokenizer conversion returned wrong length")
    table = token_table_from_strings(decoded, metadata={
        "converter": getattr(converter, "__module__", "") + "." + getattr(converter, "__name__", ""),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name": str(getattr(tokenizer, "name_or_path", "")),
    })
    table.build_seconds = time.perf_counter() - started
    return table


@dataclass
class EncodedBatch:
    packed: bytes
    offsets: np.ndarray
    status: np.ndarray

    def row(self, index: int) -> bytes:
        if self.status[index]:
            raise ValueError(f"row {index} requires native fallback, status={self.status[index]}")
        return self.packed[int(self.offsets[index]):int(self.offsets[index + 1])]

    @property
    def fallback_rows(self) -> np.ndarray:
        return np.flatnonzero(self.status)


def compile_encoder(build_dir: str | Path | None = None) -> tuple[Path, dict]:
    """Compile once by source fingerprint; record actual startup cost separately."""
    source = Path(__file__).with_suffix(".cpp")
    compiler = os.environ.get("CXX", "g++")
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "-fPIC", "-shared"]
    digest = hashlib.sha256(source.read_bytes() + repr((compiler, flags, platform.machine())).encode()).hexdigest()
    directory = Path(build_dir) if build_dir is not None else source.parent / ".build"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ("cpu_encoder_" + digest[:16] + ".so")
    command = [compiler, *flags, str(source), "-o", str(target)]
    started = time.perf_counter()
    with (directory / "cpu_encoder.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        compiled = not target.exists()
        if compiled:
            temporary = target.with_suffix(f".{os.getpid()}.tmp.so")
            try:
                subprocess.run([*command[:-1], str(temporary)], check=True,
                               capture_output=True, text=True)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return target, {"compiled": compiled, "compile_or_cache_seconds": time.perf_counter() - started,
                    "command": command, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


_I64 = C.POINTER(C.c_int64)
_F32 = C.POINTER(C.c_float)
_U64 = C.POINTER(C.c_uint64)
_U32 = C.POINTER(C.c_uint32)
_U8 = C.POINTER(C.c_uint8)


class CPUEncoder:
    def __init__(self, table: TokenTable, *, library: str | Path | None = None,
                 build_dir: str | Path | None = None):
        started = time.perf_counter()
        self.table = table
        self._lock = threading.Lock()
        self._handle = None
        if library is None:
            library, self.build_info = compile_encoder(build_dir)
        else:
            self.build_info = {"compiled": False, "library": str(library)}
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.lib.f_cpu_encoder_abi.restype = C.c_uint32
        if self.lib.f_cpu_encoder_abi() != 1:
            raise RuntimeError("encoder ABI mismatch")
        self.lib.f_cpu_encoder_create.argtypes = [C.c_char_p, C.c_uint64, _U64, _U32, _U8, C.c_uint64]
        self.lib.f_cpu_encoder_create.restype = C.c_void_p
        self.lib.f_cpu_encoder_encode.argtypes = [C.c_void_p, _I64, _F32, C.c_uint64,
            C.c_uint64, C.c_uint64, C.POINTER(C.c_void_p), C.POINTER(C.c_uint64),
            C.POINTER(_U64), C.POINTER(_U8)]
        self.lib.f_cpu_encoder_encode.restype = C.c_int
        self.lib.f_cpu_encoder_error.argtypes = [C.c_void_p]
        self.lib.f_cpu_encoder_error.restype = C.c_char_p
        self.lib.f_cpu_encoder_destroy.argtypes = [C.c_void_p]
        self.lib.f_cpu_encoder_destroy.restype = None
        self._handle = self.lib.f_cpu_encoder_create(table.blob, len(table.blob),
            table.offsets.ctypes.data_as(_U64), table.lengths.ctypes.data_as(_U32),
            table.unsafe.ctypes.data_as(_U8), table.vocab_size)
        if not self._handle:
            raise RuntimeError(self.lib.f_cpu_encoder_error(None).decode())
        self.initialization_seconds = time.perf_counter() - started

    def encode(self, token_ids: np.ndarray, logprobs: np.ndarray, *,
               top_k: int = 20) -> EncodedBatch:
        # Hidden coercion/copy would unfairly hide transfer and packing costs.
        if not isinstance(token_ids, np.ndarray) or token_ids.dtype != np.int64:
            raise TypeError("token_ids must be an int64 numpy array")
        if not isinstance(logprobs, np.ndarray) or logprobs.dtype != np.float32:
            raise TypeError("logprobs must be a float32 numpy array")
        if token_ids.ndim != 2 or token_ids.shape != logprobs.shape:
            raise ValueError("inputs must have the same [rows,columns] shape")
        if not token_ids.flags.c_contiguous or not logprobs.flags.c_contiguous:
            raise ValueError("inputs must be C-contiguous; charge packing at the caller")
        rows, columns = token_ids.shape
        if not 1 <= columns <= 256 or not 0 <= top_k <= columns:
            raise ValueError("invalid number of columns/top_k")
        packed = C.c_void_p()
        size = C.c_uint64()
        offsets = _U64()
        status = _U8()
        with self._lock:
            if not self._handle:
                raise RuntimeError("encoder closed")
            result = self.lib.f_cpu_encoder_encode(self._handle,
                token_ids.ctypes.data_as(_I64), logprobs.ctypes.data_as(_F32),
                rows, columns, top_k, C.byref(packed), C.byref(size),
                C.byref(offsets), C.byref(status))
            if result:
                raise RuntimeError(self.lib.f_cpu_encoder_error(self._handle).decode())
            return EncodedBatch(C.string_at(packed, size.value) if size.value else b"",
                np.ctypeslib.as_array(offsets, shape=(rows + 1,)).copy(),
                np.ctypeslib.as_array(status, shape=(rows,)).copy() if rows else np.empty(0, np.uint8))

    def close(self):
        with self._lock:
            if self._handle:
                self.lib.f_cpu_encoder_destroy(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_handle", None):
            self.close()
