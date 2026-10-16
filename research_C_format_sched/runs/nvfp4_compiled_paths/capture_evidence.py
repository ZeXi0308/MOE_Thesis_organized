"""CPU-only SASS extraction for the exact saved NVFP4 B1 profiler symbol.

No torch/vLLM imports, CUDA initialization, model loading or recompilation.
Only the existing vLLM main extension identified by its installed import path
is inspected. A missing/ambiguous function is a failed static verification.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def source_evidence(path, function, needles):
    text = path.read_text()
    nodes = [node for node in ast.walk(ast.parse(text))
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function]
    assert len(nodes) == 1, (str(path), function, len(nodes))
    node = nodes[0]
    lines = text.splitlines()
    selected = []
    for index in range(node.lineno - 1, node.end_lineno):
        if any(needle in lines[index] for needle in needles):
            selected.append({"line": index + 1, "text": lines[index]})
    assert selected, (str(path), needles)
    return {"path": str(path), "sha256": sha256(path), "function": function,
            "function_start_line": node.lineno, "function_end_line": node.end_lineno,
            "selected_lines": selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/research_C_format_sched"))
    parser.add_argument("--vllm", type=Path, default=Path("/root/miniconda3/lib/python3.12/site-packages/vllm"))
    parser.add_argument("--cuobjdump", default="/usr/local/cuda/bin/cuobjdump")
    args = parser.parse_args()
    root, package = args.root.resolve(), args.vllm.resolve()
    output = root / "runs/nvfp4_compiled_paths"
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "evidence.json").exists(), "Keep existing evidence; do not overwrite a completed extraction"
    record_path = root / "runs/qwen3_nvfp4_survey_r02/qwen3_uniform_nvfp4.jsonl"
    trace_path = root / "runs/qwen3_nvfp4_survey_r02/qwen3_uniform_nvfp4_traces/qwen3_nvfp4_uniform_1.json"
    records = [json.loads(line) for line in record_path.read_text().splitlines() if line.strip()]
    matches = [r for r in records if r["tokens"] == 1 and r["format"] == "nvfp4" and r["status"] == "ok"]
    assert len(matches) == 1
    record = matches[0]
    assert record["routing"]["kind"] == "uniform"
    assert record["backend"]["requested_backend"] == "cutlass"
    events = [e for e in json.loads(trace_path.read_text())["traceEvents"]
              if e.get("cat") == "kernel" and "MainloopSm120ArrayTma" in e.get("name", "")]
    symbols = {event["name"] for event in events}
    assert len(symbols) == 1, "Profiler must identify exactly one specialization"
    symbol = next(iter(symbols))
    assert symbol.startswith("_Z") and symbol in record["profile"]["kernel_names"]
    library = package / "_C_stable_libtorch.abi3.so"
    assert library.is_file()
    evidence = {"schema_version": 1, "state": "preparing", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Static SM120 instructions for the exact mangled specialization observed in the saved actual profiler trace; not dynamic hardware instruction counts",
        "library": str(library), "library_sha256": sha256(library), "library_bytes": library.stat().st_size,
        "library_identity_scope": "Installed library at CPU extraction time; the original profiler did not hash the shared library, so an unchanged historical binary is not independently proven",
        "source_record": str(record_path.relative_to(root)), "source_sha256": sha256(record_path),
        "profiler_trace": str(trace_path.relative_to(root)), "profiler_sha256": sha256(trace_path),
        "model": "qwen3", "routing": "uniform", "tokens": 1, "format": "nvfp4", "backend": "cutlass",
        "checkpoint": record["backend"]["checkpoint"], "checkpoint_revision": record["backend"]["checkpoint_revision"],
        "artifact_sha256": record["backend"]["artifact_sha256"], "profiled_kernel": symbol,
        "mangled_symbol": symbol, "mangled_symbol_characters": len(symbol),
        "profiler_kernel_event_occurrences": len(events), "profiler_and_record_exact_name_match": True,
        "architecture": "sm_120", "cpu_only": True, "gpu_work_launched": False,
        "extraction_script_sha256": sha256(__file__),
        "official_opcode_reference": {"url": "https://docs.nvidia.com/cuda/archive/13.0.3/cuda-binary-utilities/index.html#blackwell-instruction-set",
            "section": "Blackwell Instruction Set, Table 8", "opcode": "OMMA",
            "description": "FP4 Matrix Multiply and Accumulate Across a Warp"}}
    try:
        evidence["entrypoint_source_chain"] = [
            source_evidence(root / "runs/qwen3_nvfp4_survey_r02/submitted_code/format_backends.py", "build_backend", ["run_cutlass_moe_fp4", "nvfp4_backend == 'cutlass'"]),
            source_evidence(package / "model_executor/layers/fused_moe/experts/cutlass_moe.py", "run_cutlass_moe_fp4", ["ops.cutlass_fp4_moe_mm"]),
            source_evidence(package / "_custom_ops.py", "cutlass_fp4_moe_mm", ["torch.ops._C.cutlass_fp4_group_mm"]),
            source_evidence(package / "platforms/cuda.py", "import_kernels", ["import vllm._C_stable_libtorch"])]
        original_hash = record["backend"]["conversion"]["vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassExpertsFp4.process_weights_after_loading"]["source_file_sha256"]
        evidence["installed_cutlass_moe_sha256_matches_recorded_conversion_source"] = evidence["entrypoint_source_chain"][1]["sha256"] == original_hash
        evidence["original_conversion_cutlass_moe_sha256"] = original_hash
        evidence["entrypoint_chain"] = "format_backends.build_backend -> run_cutlass_moe_fp4 -> ops.cutlass_fp4_moe_mm -> torch.ops._C.cutlass_fp4_group_mm; CUDA platform imports vllm._C_stable_libtorch"
        command = [args.cuobjdump, "--gpu-architecture", "sm_120", "--function", symbol, "--dump-sass", str(library)]
        evidence["command"] = command
        version = subprocess.run([args.cuobjdump, "--version"], capture_output=True, text=True, check=True, timeout=20)
        evidence["cuobjdump_version"] = version.stdout.strip()
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (output / "profiled_b1_sm120.sass").write_text(result.stdout)
        (output / "cuobjdump.stderr.log").write_text(result.stderr)
        (output / "profiled_mangled_symbol.txt").write_text(symbol + "\n")
        evidence["cuobjdump_returncode"] = result.returncode
        result.check_returncode()
        functions = re.findall(r"^\s*Function :\s*(\S+)\s*$", result.stdout, re.MULTILINE)
        assert functions == [symbol], (len(functions), "exact single-function match required")
        architectures = set(re.findall(r"^arch = (\S+)\s*$", result.stdout, re.MULTILINE))
        assert architectures == {"sm_120"}, architectures
        instruction_lines = [line.strip() for line in result.stdout.splitlines() if re.search(r"\bOMMA\.", line)]
        assert instruction_lines, "No FP4 warp matrix multiply instruction in this exact compiled function"
        opcodes = Counter(re.search(r"\bOMMA\.[A-Z0-9.]+", line).group(0) for line in instruction_lines)
        assert "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X" in opcodes
        assert sha256(library) == evidence["library_sha256"], "Library changed during extraction"
        evidence.update(state="complete", exact_single_function_match=True, function_count=1,
            tensor_core_instruction="OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X",
            static_instruction_occurrences=len(instruction_lines), static_opcode_counts=dict(opcodes),
            instruction_lines=instruction_lines,
            interpretation="The actual profiler kernel name exactly matches one SM120 function containing block-scaled FP4 E2M1 x E2M1 warp matrix multiply instructions with F32 accumulation; supports the FP4 Tensor Core path classification for this observed specialization",
            limitations=["Static occurrences are not dynamic executed instruction counts, Tensor Core utilization, occupancy or achieved throughput",
                "The two profiler launches share the same specialization; instruction occurrences must not be multiplied by launch count to estimate dynamic work",
                "No new GPU measurement was performed; this does not establish a performance or energy cause by itself"],
            stderr_note="cuobjdump warns for fatbin members without the requested symbol; stdout contains exactly one matching function and only SM120 architecture headers")
    except BaseException as error:
        evidence.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        evidence["ended_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        hashes = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name not in ("file_hashes.json", "local_backup_receipt.json")}
        (output / "file_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({"state": evidence["state"], "library_sha256": evidence["library_sha256"],
        "exact_single_function_match": evidence["exact_single_function_match"],
        "static_opcode_counts": evidence["static_opcode_counts"], "files": len(hashes)}))


if __name__ == "__main__":
    main()
