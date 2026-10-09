"""Small stdlib-only ABI test; native vLLM semantic comparison is separate.

Run: python3 cpu_encoder_test.py
This is correctness validation, not a performance or native-frontend benchmark.
"""
import ctypes as C
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import tempfile


def main():
    source = Path(__file__).with_name("cpu_encoder.cpp")
    with tempfile.TemporaryDirectory(prefix="f-cpu-encoder-test-") as tmp:
        library = Path(tmp) / "encoder.so"
        subprocess.run(["g++", "-std=c++17", "-O3", "-fPIC", "-shared",
                        str(source), "-o", str(library)], check=True)
        lib = C.CDLL(str(library))
        u64, u32, u8 = C.POINTER(C.c_uint64), C.POINTER(C.c_uint32), C.POINTER(C.c_uint8)
        lib.f_cpu_encoder_create.argtypes = [C.c_char_p, C.c_uint64, u64, u32, u8, C.c_uint64]
        lib.f_cpu_encoder_create.restype = C.c_void_p
        lib.f_cpu_encoder_encode.argtypes = [C.c_void_p, C.POINTER(C.c_int64), C.POINTER(C.c_float),
            C.c_uint64, C.c_uint64, C.c_uint64, C.POINTER(C.c_void_p), C.POINTER(C.c_uint64),
            C.POINTER(u64), C.POINTER(u8)]
        lib.f_cpu_encoder_encode.restype = C.c_int
        lib.f_cpu_encoder_destroy.argtypes = [C.c_void_p]
        tokens = [' a', 'quote"', '\n', '汉🙂', '', '�']
        blob, off, length = bytearray(), [], []
        for token in tokens:
            for piece in [b'{"token":' + json.dumps(token, ensure_ascii=False).encode() + b',"logprob":',
                          b',"bytes":' + json.dumps(list(token.encode()), separators=(',', ':')).encode()]:
                off.append(len(blob)); length.append(len(piece)); blob.extend(piece)
        handle = lib.f_cpu_encoder_create(bytes(blob), len(blob),
            (C.c_uint64 * len(off))(*off), (C.c_uint32 * len(length))(*length),
            (C.c_uint8 * len(tokens))(0, 0, 0, 0, 0, 1), len(tokens))
        assert handle
        try:
            rng = random.Random(92331)
            finite = [-0.0, 0.0, -float('inf'), -10000.0, -0.1, -1e-40, 1e38]
            while len(finite) < 4007:
                x = struct.unpack('f', struct.pack('I', rng.getrandbits(32)))[0]
                if math.isfinite(x): finite.append(x)
            rows = []
            for x in finite:
                rows.append(([0, 1, 0, 2, 3, 4], [-7.0, -2.25, x, -3.0, -4.0, -5.0]))
            rows.extend([([0, 1, 2, 3, 4, 5], [-1.0] * 6),
                         ([0, 1, 2, 3, 4, 1000], [-1.0] * 6),
                         ([0, 1, 2, 3, 4, 0], [float('nan')] * 6),
                         ([0, 1, 2, 3, 4, 0], [float('inf')] * 6)])
            ids = (C.c_int64 * (len(rows) * 6))(*(x for row, _ in rows for x in row))
            probs = (C.c_float * (len(rows) * 6))(*(x for _, row in rows for x in row))
            packed, size, offsets, statuses = C.c_void_p(), C.c_uint64(), u64(), u8()
            assert lib.f_cpu_encoder_encode(handle, ids, probs, len(rows), 6, 4,
                C.byref(packed), C.byref(size), C.byref(offsets), C.byref(statuses)) == 0
            output = C.string_at(packed, size.value)
            for i, (row, ps) in enumerate(rows[:-4]):
                assert statuses[i] == 0
                value = json.loads(output[offsets[i]:offsets[i + 1]])
                # Independent oracle: normal Python dict semantics plus normal
                # JSON serialization of API-shaped scalar/byte fields.
                entries = {}
                for token_id, lp in zip(row, ps):
                    entries[token_id] = {'token': tokens[token_id],
                        'logprob': max(float(C.c_float(lp).value), -9999.0),
                        'bytes': list(tokens[token_id].encode())}
                expected = dict(entries[row[0]])
                expected['top_logprobs'] = list(entries.values())[:4]
                assert value == expected, (i, value, expected)
                assert isinstance(value['logprob'], float)
                assert math.copysign(1.0, value['logprob']) == math.copysign(1.0, expected['logprob'])
            assert [statuses[len(rows) - 4 + j] for j in range(4)] == [1, 2, 3, 3]
            assert all(offsets[i] == offsets[i + 1] for i in range(len(rows) - 4, len(rows)))
            print(json.dumps({'ok': True, 'encoded_rows': len(rows) - 4,
                              'fallback_rows': 4, 'bytes': size.value,
                              'native_vllm_validation': False}))
        finally:
            lib.f_cpu_encoder_destroy(handle)


if __name__ == '__main__':
    main()
