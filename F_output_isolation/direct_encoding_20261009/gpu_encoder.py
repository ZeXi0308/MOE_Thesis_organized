"""CUDA array-to-JSON prototype for one already-ready sampling batch.

Import and compilation do not initialize CUDA. GPUEncoder construction does:
the caller must hold the research GPU lock before constructing it or passing
CUDA tensors. The token table and EncodedBatch ABI are shared with cpu_encoder.

The fast path has three kernels: per-candidate exact-float formatting and row
layout, a row offset scan, and parallel writing directly into packed output.
It never waits for a future token. Unsafe token rows are empty with a fallback
status; the caller must execute the real native compatibility path and charge
that work, rather than omitting their output.

encode_device() consumes contiguous resident CUDA arrays, synchronizes the
specified stream, copies actual output bytes to pinned host storage, then
copies packed bytes/metadata into independent Python-owned objects. It does
not hide an H2D upload or CPU input pack; a replay caller must charge those
separately. Queueing on the specified stream IS included in wall time. Kernel
CUDA-event timing alone is not an end-to-end latency or co-run interference
measurement. Residency and one-time initialization are separately reported.

Float strings are exact decimal expansions of float32, hence can be longer
than CPU shortest-round-trip strings. This extra D2H, frontend/network and
client parsing cost must stay in the comparison. Values are not quantized.
"""
from __future__ import annotations

import argparse
import ctypes as C
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np

from cpu_encoder import EncodedBatch, TokenTable


def compile_encoder(build_dir: str | Path | None = None, *,
                    nvcc: str = "/usr/local/cuda/bin/nvcc",
                    arch: str = "sm_120") -> tuple[Path, dict]:
    """Explicit compile, with source fingerprint and no model/GPU launch."""
    source = Path(__file__).with_suffix(".cu")
    cuda_root = Path(nvcc).resolve().parent.parent
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "--shared", "-Xcompiler", "-fPIC",
             f"-arch={arch}", "-Xlinker", "-rpath", "-Xlinker", str(cuda_root / "lib64")]
    digest = hashlib.sha256(source.read_bytes() + repr((nvcc, flags)).encode()).hexdigest()
    directory = Path(build_dir) if build_dir is not None else source.parent / ".build"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ("gpu_encoder_" + digest[:16] + ".so")
    command = [nvcc, *flags, str(source), "-o", str(target)]
    started = time.perf_counter()
    with (directory / "gpu_encoder.lock").open("a") as lock:
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


_U64 = C.POINTER(C.c_uint64)
_U32 = C.POINTER(C.c_uint32)
_U8 = C.POINTER(C.c_uint8)


class GPUEncoder:
    """Persistent CUDA context with the same token-fragment cache as CPUEncoder.

    max_rows reserves output capacity; exceeding it raises rather than silently
    splitting/rebatching. Resident and pinned output buffers use a conservative
    vocabulary-wide bound and are fully disclosed in initialization_info.
    """
    def __init__(self, table: TokenTable, *, max_rows: int = 256, device: int = 0,
                 library: str | Path | None = None, build_dir: str | Path | None = None,
                 nvcc: str = "/usr/local/cuda/bin/nvcc", arch: str = "sm_120"):
        started = time.perf_counter()
        self._handle = None
        self._lock = threading.Lock()
        self.table = table
        self.device = int(device)
        self.max_rows = int(max_rows)
        self.last_metrics: dict = {}
        if library is None:
            library, self.build_info = compile_encoder(build_dir, nvcc=nvcc, arch=arch)
        else:
            self.build_info = {"compiled": False, "library": str(library)}
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.lib.f_gpu_error.argtypes = []
        self.lib.f_gpu_error.restype = C.c_char_p
        self.lib.f_gpu_create.argtypes = [C.c_char_p, C.c_uint64, _U64, _U32, _U8,
                                         C.c_uint64, C.c_int, C.c_int]
        self.lib.f_gpu_create.restype = C.c_void_p
        self.lib.f_gpu_destroy.argtypes = [C.c_void_p]
        self.lib.f_gpu_destroy.restype = None
        self.lib.f_gpu_device_bytes.argtypes = [C.c_void_p]
        self.lib.f_gpu_device_bytes.restype = C.c_uint64
        self.lib.f_gpu_pinned_bytes.argtypes = [C.c_void_p]
        self.lib.f_gpu_pinned_bytes.restype = C.c_uint64
        self.lib.f_gpu_encode.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p,
            C.c_int, C.c_int, C.c_int, C.c_uint64, C.POINTER(C.c_void_p),
            C.POINTER(_U64), C.POINTER(_U8), C.POINTER(C.c_uint64),
            C.POINTER(C.c_float), C.POINTER(C.c_float)]
        self.lib.f_gpu_encode.restype = C.c_int
        self._handle = self.lib.f_gpu_create(table.blob, len(table.blob),
            table.offsets.ctypes.data_as(_U64), table.lengths.ctypes.data_as(_U32),
            table.unsafe.ctypes.data_as(_U8), table.vocab_size, self.max_rows, self.device)
        if not self._handle:
            raise RuntimeError(self.lib.f_gpu_error().decode())
        self.initialization_seconds = time.perf_counter() - started
        self.initialization_info = {
            "device": self.device, "max_rows": self.max_rows,
            "table_nbytes": table.nbytes,
            "device_reserved_bytes": int(self.lib.f_gpu_device_bytes(self._handle)),
            "pinned_host_reserved_bytes": int(self.lib.f_gpu_pinned_bytes(self._handle)),
            "initialization_seconds": self.initialization_seconds,
            "float_encoding": "exact_decimal_binary32_promoted_to_binary64",
            "compile": self.build_info,
        }

    def encode_device(self, token_ids, logprobs, *, top_k: int = 20,
                      stream=None) -> EncodedBatch:
        """Encode CUDA torch tensors; no implicit conversion, pack, or upload.

        stream may be a torch CUDA Stream, integer cudaStream_t pointer, or None
        for the current torch stream. Dependencies from other producer streams
        are the caller's responsibility, exactly as with a native CUDA kernel.
        """
        started = time.perf_counter()
        cpu_started = time.process_time()
        import torch
        if not isinstance(token_ids, torch.Tensor) or not isinstance(logprobs, torch.Tensor):
            raise TypeError("encode_device requires torch CUDA tensors")
        if token_ids.dtype not in (torch.int32, torch.int64) or logprobs.dtype != torch.float32:
            raise TypeError("token_ids must be int32/int64; logprobs must be float32")
        if not token_ids.is_cuda or not logprobs.is_cuda:
            raise ValueError("inputs must already reside on CUDA; charge uploads explicitly")
        if token_ids.device != logprobs.device or token_ids.device.index != self.device:
            raise ValueError("input CUDA device differs from encoder device")
        if token_ids.ndim != 2 or token_ids.shape != logprobs.shape:
            raise ValueError("inputs must have matching [rows,columns] shapes")
        if not token_ids.is_contiguous() or not logprobs.is_contiguous():
            raise ValueError("inputs must be contiguous; charge any packing at the caller")
        rows, columns = token_ids.shape
        if not 0 <= rows <= self.max_rows or not 1 <= columns <= 33:
            raise ValueError("row capacity exceeded or columns outside [1,33]")
        if not 0 <= top_k <= min(columns, 32):
            raise ValueError("top_k must be in [0,min(columns,32)]")
        if rows == 0:
            result = EncodedBatch(b"", np.zeros(1, np.uint64), np.zeros(0, np.uint8))
            self.last_metrics = {"total_wall_s": time.perf_counter() - started,
                                 "host_cpu_s": time.process_time() - cpu_started,
                                 "kernel_ms": 0.0, "d2h_event_ms": 0.0,
                                 "rows": 0, "packed_bytes": 0, "fallback_rows": 0}
            return result
        if stream is None:
            stream_ptr = int(torch.cuda.current_stream(token_ids.device).cuda_stream)
        elif isinstance(stream, int):
            stream_ptr = stream
        else:
            stream_ptr = int(stream.cuda_stream)
        packed = C.c_void_p()
        offsets = _U64()
        status = _U8()
        packed_bytes = C.c_uint64()
        kernel_ms, d2h_ms = C.c_float(), C.c_float()
        with self._lock:
            if not self._handle:
                raise RuntimeError("encoder closed")
            error = self.lib.f_gpu_encode(self._handle,
                C.c_void_p(token_ids.data_ptr()), token_ids.element_size(),
                C.c_void_p(logprobs.data_ptr()), rows, columns, top_k, stream_ptr,
                C.byref(packed), C.byref(offsets), C.byref(status), C.byref(packed_bytes),
                C.byref(kernel_ms), C.byref(d2h_ms))
            if error:
                raise RuntimeError(self.lib.f_gpu_error().decode())
            result = EncodedBatch(C.string_at(packed, packed_bytes.value),
                np.ctypeslib.as_array(offsets, shape=(rows + 1,)).copy(),
                np.ctypeslib.as_array(status, shape=(rows,)).copy())
            self.last_metrics = {
                "total_wall_s": time.perf_counter() - started,
                "host_cpu_s": time.process_time() - cpu_started,
                "kernel_ms": float(kernel_ms.value), "d2h_event_ms": float(d2h_ms.value),
                "rows": rows, "columns": columns, "top_k": top_k,
                "packed_bytes": int(packed_bytes.value),
                "metadata_d2h_bytes": (rows + 1) * 8 + rows,
                "fallback_rows": int(np.count_nonzero(result.status)),
                "input_transfer_included": False,
            }
            return result

    encode = encode_device

    def close(self) -> None:
        with self._lock:
            if self._handle:
                self.lib.f_gpu_destroy(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        if getattr(self, "_handle", None):
            self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile the CUDA encoder without starting GPU work")
    parser.add_argument("--build", action="store_true", required=True)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    parser.add_argument("--arch", default="sm_120")
    args = parser.parse_args()
    library, info = compile_encoder(args.build_dir, nvcc=args.nvcc, arch=args.arch)
    print(json.dumps({"library": str(library), **info}, indent=2))
