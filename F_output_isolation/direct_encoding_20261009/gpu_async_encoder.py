"""Per-request asynchronous GPU encoder and shared compact CPU transport.

This COMPONENT must be hooked while sampler arrays are still CUDA tensors,
before their first D2H. No CPU/numpy -> GPU upload is accepted. Import/build do
not initialize CUDA; constructing GPUAsyncEncoder does, under the caller's GPU
lock. The caller submits only the already-ready rows of ONE request.

API:
  ticket = encoder.submit_device(ids, values, ranks, producer_stream, encode=True)
  # CompatibilityRequired: immediately use the native/CPU compatibility path.
  for ticket in encoder.poll_ready():
      result = encoder.collect(ticket)  # Python-owned arrays/bytes, never waits
      encoder.release(ticket)
      request_future[ticket.handle].set_result(result)

encode=False is the strong CPU transport control: same ring, producer-stream
D2D snapshot/lifetime, event dependency and polling; three compact D2H copies,
no encoding kernel or expanded padding. result.encoded is None and raw arrays
retain their captured/sampler dtypes. The CPU worker must charge its required
int64 conversion and direct encoding after this result becomes ready.

IMPORTANT lifetime contract: producer_stream is the stream that owns writes
to these arrays. submit queues sampling_ready, the D2D snapshot and
snapshot_ready ON THAT STREAM before returning. A subsequent overwrite on the
same stream is ordered after the snapshot. Concurrent writes from another
stream are not permitted without an explicit dependency. Strong references
alone would not protect reused graph buffers. Snapshot cost/producer ordering
must remain in the real next-inference-step comparison.

Each request is encoded on one of a fixed stream pool. There is no all-request
completion barrier; requests sharing a stream still experience its normal
queueing. All fixed-slot padding and raw fallback data cross D2H and are
reported, including initialization of slot padding by the kernel. poll uses
cudaEventQuery only. No submit/collect/release waits or CUDA allocations occur.
Python bookkeeping/output ownership allocations ARE charged to CPU timings.

The service owns per-request ordering/cancellation. Cancellation can discard
the result but must poll and release the ticket after completion. close refuses
outstanding tickets. Initialization/destruction allocate/free resources and
are outside the hot path; runtime allocator cleanup may itself synchronize.
This module neither starts polling threads nor changes admission/generation.
"""
from __future__ import annotations

import argparse
import ctypes as C
from dataclasses import dataclass, field
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


def compile_encoder(build_dir=None, *, nvcc="/usr/local/cuda/bin/nvcc", arch="sm_120"):
    source = Path(__file__).with_suffix(".cu")
    cuda_root = Path(nvcc).resolve().parent.parent
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "--shared", "-Xcompiler", "-fPIC",
             f"-arch={arch}", "-Xlinker", "-rpath", "-Xlinker", str(cuda_root / "lib64")]
    digest = hashlib.sha256(source.read_bytes() + repr((nvcc, flags)).encode()).hexdigest()
    directory = Path(build_dir) if build_dir else source.parent / ".build"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ("gpu_async_encoder_" + digest[:16] + ".so")
    command = [nvcc, *flags, str(source), "-o", str(target)]
    started = time.perf_counter()
    with (directory / "gpu_async_encoder.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        compiled = not target.exists()
        if compiled:
            temporary = target.with_suffix(f".{os.getpid()}.tmp.so")
            try:
                subprocess.run([*command[:-1], str(temporary)], check=True, capture_output=True, text=True)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return target, {"compiled": compiled, "compile_or_cache_seconds": time.perf_counter()-started,
                    "command": command, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


@dataclass(frozen=True)
class CompatibilityRequired:
    reason: str
    rows: int
    submit_wall_s: float
    submit_cpu_s: float
    accepted: bool = field(default=False, init=False)

    def __bool__(self):
        return False


@dataclass(frozen=True)
class GPUAsyncTicket:
    handle: int
    rows: int
    encode: bool
    submitted_at: float
    submit_wall_s: float
    submit_cpu_s: float
    producer_stream_ptr: int
    _owner: int = field(repr=False)
    accepted: bool = field(default=True, init=False)


@dataclass
class CollectedRequest:
    encoded: EncodedBatch | None
    raw_ids: np.ndarray
    raw_values: np.ndarray
    raw_ranks: np.ndarray
    metrics: dict

    @property
    def sampled_ids(self):
        return self.raw_ids[:, 0]

    @property
    def sampled_values(self):
        return self.raw_values[:, 0]


class _RowHeader(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ("status", "json_length", "columns", "reserved")] + [
        ("rank", C.c_int64), ("ids", C.c_int64 * 21), ("values", C.c_float * 21)]


class _Metrics(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in
        ("rows", "snapshot_bytes", "transferred_bytes", "header_bytes", "valid_json_bytes")] + [
        (name, C.c_float) for name in
        ("snapshot_ms", "kernel_ms", "d2h_ms", "sampling_ready_to_host_ms")] + [
        (name, C.c_uint32) for name in ("encoded", "id_bytes", "rank_bytes", "reserved")]


_U64, _U32, _U8 = C.POINTER(C.c_uint64), C.POINTER(C.c_uint32), C.POINTER(C.c_uint8)


class GPUAsyncEncoder:
    def __init__(self, table: TokenTable, *, max_inflight=32, max_rows_per_request=4,
                 num_streams=4, device=0, library=None, build_dir=None,
                 nvcc="/usr/local/cuda/bin/nvcc", arch="sm_120"):
        started = time.perf_counter()
        self._handle = None
        self._lock = threading.Lock()
        self._tickets = {}
        self._collected = {}
        self.table, self.device = table, int(device)
        self.max_inflight, self.max_rows_per_request = int(max_inflight), int(max_rows_per_request)
        self.stats = {"submitted": 0, "compatibility": 0, "poll_calls": 0,
                      "poll_wall_s": 0.0, "poll_cpu_s": 0.0,
                      "released": 0, "release_wall_s": 0.0, "release_cpu_s": 0.0}
        if library is None:
            library, self.build_info = compile_encoder(build_dir, nvcc=nvcc, arch=arch)
        else:
            self.build_info = {"compiled": False, "library": str(library)}
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.lib.f_async_error.restype = C.c_char_p
        self.lib.f_async_error.argtypes = []
        self.lib.f_async_abi.restype = C.c_uint32
        self.lib.f_async_abi.argtypes = []
        if self.lib.f_async_abi() != 1:
            raise RuntimeError("async encoder ABI mismatch")
        self.lib.f_async_create.argtypes = [C.c_char_p, C.c_uint64, _U64, _U32, _U8,
                                           C.c_uint64, C.c_int, C.c_int, C.c_int, C.c_int]
        self.lib.f_async_create.restype = C.c_void_p
        self.lib.f_async_submit.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p,
            C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_uint64, C.POINTER(C.c_uint64)]
        self.lib.f_async_submit.restype = C.c_int
        for name in ("f_async_poll", "f_async_release"):
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = [C.c_void_p, C.c_uint64], C.c_int
        self.lib.f_async_collect_view.argtypes = [C.c_void_p, C.c_uint64, C.POINTER(C.c_void_p),
                                                C.POINTER(C.c_uint64), C.POINTER(_Metrics)]
        self.lib.f_async_collect_view.restype = C.c_int
        self.lib.f_async_destroy.argtypes = [C.c_void_p]
        self.lib.f_async_destroy.restype = C.c_int
        for name in ("f_async_device_bytes", "f_async_pinned_bytes", "f_async_row_stride"):
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = [C.c_void_p], C.c_uint64
        self.lib.f_async_header_bytes.argtypes = []
        self.lib.f_async_header_bytes.restype = C.c_uint64
        if self.lib.f_async_header_bytes() != C.sizeof(_RowHeader):
            raise RuntimeError("row header layout mismatch")
        self._handle = self.lib.f_async_create(table.blob, len(table.blob),
            table.offsets.ctypes.data_as(_U64), table.lengths.ctypes.data_as(_U32),
            table.unsafe.ctypes.data_as(_U8), table.vocab_size, self.max_inflight,
            self.max_rows_per_request, int(num_streams), self.device)
        if not self._handle:
            raise RuntimeError(self.lib.f_async_error().decode())
        self.initialization_info = {
            "initialization_s": time.perf_counter()-started, "build": self.build_info,
            "device": self.device, "max_inflight": self.max_inflight,
            "max_rows_per_request": self.max_rows_per_request, "num_streams": int(num_streams),
            "table_nbytes": table.nbytes,
            "device_reserved_bytes": int(self.lib.f_async_device_bytes(self._handle)),
            "pinned_host_reserved_bytes": int(self.lib.f_async_pinned_bytes(self._handle)),
            "row_slot_bytes": int(self.lib.f_async_row_stride(self._handle)),
            "row_header_bytes": C.sizeof(_RowHeader),
            "snapshot_stream": "producer; ordered before next same-stream overwrite",
        }

    def _error(self):
        return RuntimeError(self.lib.f_async_error().decode())

    def _validate_ticket(self, ticket):
        if not self._handle:
            raise RuntimeError("encoder closed")
        if not isinstance(ticket, GPUAsyncTicket) or ticket._owner != id(self):
            raise ValueError("ticket belongs to another encoder")
        if ticket.handle not in self._tickets or self._tickets[ticket.handle][0] != ticket:
            raise ValueError("stale/released ticket")

    def submit_device(self, ids, values, ranks, producer_stream, *, encode=True):
        started, cpu_start = time.perf_counter(), time.thread_time()
        import torch
        arrays = (ids, values, ranks)
        if not all(isinstance(x, torch.Tensor) and x.is_cuda for x in arrays):
            raise TypeError("submit_device accepts CUDA tensors only; no numpy/H2D staging")
        if any(x.device != ids.device for x in arrays) or ids.device.index != self.device:
            raise ValueError("request arrays must be on the encoder's device")
        if ids.dtype not in (torch.int32, torch.int64) or values.dtype != torch.float32:
            raise TypeError("IDs must be int32/int64; values float32")
        if ranks.dtype not in (torch.int32, torch.int64):
            raise TypeError("ranks must be int32/int64")
        if ids.ndim != 2 or ids.shape[1] != 21 or values.shape != ids.shape:
            raise ValueError("one request's sampled+top20 rows must have shape [rows,21]")
        rows = ids.shape[0]
        if ranks.ndim != 1 or ranks.shape[0] != rows or rows < 1:
            raise ValueError("ranks must have shape [rows], with at least one already-ready row")
        if any(not x.is_contiguous() for x in arrays):
            raise ValueError("arrays must be contiguous; do not hide a pack inside submit")
        if producer_stream is None:
            raise ValueError("explicit producer_stream is required for input lifetime ordering")
        stream_ptr = int(producer_stream) if isinstance(producer_stream, int) else int(producer_stream.cuda_stream)
        raw_ticket = C.c_uint64()
        with self._lock:
            if not self._handle:
                raise RuntimeError("encoder closed")
            result = self.lib.f_async_submit(self._handle, C.c_void_p(ids.data_ptr()), ids.element_size(),
                C.c_void_p(values.data_ptr()), C.c_void_p(ranks.data_ptr()), ranks.element_size(),
                rows, int(bool(encode)), stream_ptr, C.byref(raw_ticket))
            if result < 0:
                raise self._error()
            wall_s, cpu_s = time.perf_counter()-started, time.thread_time()-cpu_start
            if result:
                reason = {1: "fixed_ticket_pool_full", 2: "request_exceeds_fixed_row_capacity",
                          3: "producer_stream_is_capturing"}[result]
                self.stats["compatibility"] += 1
                return CompatibilityRequired(reason, rows, wall_s, cpu_s)
            ticket = GPUAsyncTicket(int(raw_ticket.value), rows, bool(encode), started, wall_s,
                                    cpu_s, stream_ptr, id(self))
            # Retain storage and producer stream objects until release. Correct
            # reuse ordering is provided by the producer-stream snapshot, not
            # these references alone.
            self._tickets[ticket.handle] = (ticket, arrays, producer_stream)
            self.stats["submitted"] += 1
            return ticket

    def poll_ready(self, ticket=None):
        started, cpu_start = time.perf_counter(), time.thread_time()
        with self._lock:
            if not self._handle:
                raise RuntimeError("encoder closed")
            tickets = [ticket] if ticket is not None else [v[0] for v in self._tickets.values()]
            completed = []
            for current in tickets:
                self._validate_ticket(current)
                result = self.lib.f_async_poll(self._handle, current.handle)
                if result < 0:
                    raise self._error()
                if result:
                    completed.append(current)
            self.stats["poll_calls"] += 1
            self.stats["poll_wall_s"] += time.perf_counter()-started
            self.stats["poll_cpu_s"] += time.thread_time()-cpu_start
            return bool(completed) if ticket is not None else completed

    def collect(self, ticket):
        """Return Python-owned data only if THIS request is ready; else None."""
        started, cpu_start = time.perf_counter(), time.thread_time()
        with self._lock:
            self._validate_ticket(ticket)
            if ticket.handle in self._collected:
                return self._collected[ticket.handle]
            address, stride, metrics = C.c_void_p(), C.c_uint64(), _Metrics()
            ready = self.lib.f_async_collect_view(self._handle, ticket.handle,
                C.byref(address), C.byref(stride), C.byref(metrics))
            if ready < 0:
                raise self._error()
            if not ready:
                return None
            count = int(metrics.rows)
            base = address.value
            if metrics.encoded:
                raw_ids = np.empty((count, 21), dtype=np.int64)
                raw_values = np.empty((count, 21), dtype=np.float32)
                raw_ranks = np.empty(count, dtype=np.int64)
                statuses, offsets = np.empty(count, np.uint8), np.zeros(count+1, np.uint64)
                pieces = []
                for row in range(count):
                    row_at = base + row*stride.value
                    header = _RowHeader.from_address(row_at)
                    raw_ids[row] = np.ctypeslib.as_array(header.ids)
                    raw_values[row] = np.ctypeslib.as_array(header.values)
                    raw_ranks[row] = header.rank
                    statuses[row] = header.status
                    piece = C.string_at(row_at+C.sizeof(_RowHeader), header.json_length)
                    pieces.append(piece)
                    offsets[row+1] = offsets[row]+len(piece)
                encoded = EncodedBatch(b"".join(pieces), offsets, statuses)
            else:
                id_dtype = np.int32 if metrics.id_bytes == 4 else np.int64
                rank_dtype = np.int32 if metrics.rank_bytes == 4 else np.int64
                ids_bytes, values_bytes = count*21*metrics.id_bytes, count*21*4
                # Copies establish independent Python ownership before release;
                # their cost is in collect_cpu_s, not hidden from CPU transport.
                raw_ids = np.frombuffer(C.string_at(base, ids_bytes), dtype=id_dtype).reshape(count,21)
                raw_values = np.frombuffer(C.string_at(base+ids_bytes, values_bytes), dtype=np.float32).reshape(count,21)
                raw_ranks = np.frombuffer(C.string_at(base+ids_bytes+values_bytes, count*metrics.rank_bytes), dtype=rank_dtype)
                encoded = None
            details = {name: getattr(metrics, name) for name, _ in _Metrics._fields_ if name != "reserved"}
            details.update({
                "submit_wall_s": ticket.submit_wall_s, "submit_cpu_s": ticket.submit_cpu_s,
                "collect_wall_s": time.perf_counter()-started,
                "collect_cpu_s": time.thread_time()-cpu_start,
                "submit_to_collect_wall_s": time.perf_counter()-ticket.submitted_at,
                "padding_d2h_bytes": int(metrics.transferred_bytes-metrics.header_bytes-metrics.valid_json_bytes)
                    if metrics.encoded else 0,
                "fallback_rows": int(np.count_nonzero(encoded.status)) if encoded is not None else None,
                "producer_snapshot_copies": 3,
                "d2h_copies": 1 if metrics.encoded else 3,
                "kernel_launches": 1 if metrics.encoded else 0,
                "cross_request_completion_barrier": False,
            })
            result = CollectedRequest(encoded, raw_ids, raw_values, raw_ranks, details)
            self._collected[ticket.handle] = result
            return result

    def release(self, ticket):
        """Release ready storage; a running ticket returns False, never waits."""
        started, cpu_start = time.perf_counter(), time.thread_time()
        with self._lock:
            self._validate_ticket(ticket)
            released = self.lib.f_async_release(self._handle, ticket.handle)
            if released < 0:
                raise self._error()
            if released:
                self._tickets.pop(ticket.handle)
                self._collected.pop(ticket.handle, None)
                self.stats["released"] += 1
            self.stats["release_wall_s"] += time.perf_counter()-started
            self.stats["release_cpu_s"] += time.thread_time()-cpu_start
            return bool(released)

    @property
    def outstanding(self):
        with self._lock:
            return len(self._tickets)

    def close(self):
        with self._lock:
            if self._handle:
                if not self.lib.f_async_destroy(self._handle):
                    raise RuntimeError("outstanding CUDA tickets: poll/drop/release them before close; close will not wait")
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        # Never free a running DMA's buffers. Explicit close reports unfinished
        # tickets; an abandoned live context is left for process-level cleanup.
        if getattr(self, "_handle", None) and not getattr(self, "_tickets", {}):
            self.lib.f_async_destroy(self._handle)
            self._handle = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile the async CUDA component; launch no GPU work")
    parser.add_argument("--build", action="store_true", required=True)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    parser.add_argument("--arch", default="sm_120")
    args = parser.parse_args()
    path, info = compile_encoder(args.build_dir, nvcc=args.nvcc, arch=args.arch)
    print(json.dumps({"library": str(path), **info}, indent=2))
