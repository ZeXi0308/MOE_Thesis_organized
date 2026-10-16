"""CPU-only extraction of the exact Marlin specialization in the saved trace."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

root = Path('/root/autodl-tmp/research_C_format_sched')
out = root / 'runs/awq_compiled_paths'
library = Path('/root/miniconda3/lib/python3.12/site-packages/vllm/_moe_C_stable_libtorch.abi3.so')
source = root / 'runs/qwen3_awq_survey_r01/qwen3_uniform_awq.jsonl'
record = json.loads(source.read_text().splitlines()[0])
profiled = next(n for n in record['profile']['kernel_names'] if 'marlin_moe_wna16::Marlin<' in n)
symbols = [line.split()[-1] for line in (out / 'sm120_symbols.txt').read_text().splitlines()
           if 'STT_FUNC' in line and 'marlin_moe_wna166Marlin' in line]
demangled = subprocess.run(['c++filt'], input='\n'.join(symbols), text=True,
                          capture_output=True, check=True).stdout.splitlines()
matches = list(dict.fromkeys(s for s, d in zip(symbols, demangled) if d == profiled))
assert len(matches) == 1, matches
command = ['/usr/local/cuda/bin/cuobjdump', '--gpu-architecture', 'sm_120',
           '--function', matches[0], '--dump-sass', str(library)]
result = subprocess.run(command, text=True, capture_output=True, check=True)
assert result.stdout.count('Function :') == 1
assert re.search(r'Function :\s*' + re.escape(matches[0]), result.stdout)
instructions = [line.strip() for line in result.stdout.splitlines() if 'HMMA.16816.F32.BF16' in line]
assert instructions
(out / 'profiled_b1_sm120.sass').write_text(result.stdout)
(out / 'cuobjdump.stderr.log').write_text(result.stderr)
evidence = {'scope': 'Static SM120 instructions for the exact specialization observed in one actual profiler trace; not dynamic hardware instruction counts',
            'library': str(library), 'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
            'source_record': str(source.relative_to(root)), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'model': 'qwen3', 'routing': 'uniform', 'tokens': record['tokens'],
            'profiled_kernel': profiled, 'mangled_symbol': matches[0],
            'demangled_name_exactly_matches_profile': True,
            'command': command, 'architecture': 'sm_120',
            'tensor_core_instruction': 'HMMA.16816.F32.BF16',
            'static_instruction_occurrences': len(instructions), 'instruction_lines': instructions,
            'stderr_note': 'cuobjdump warns for other fatbin members without this specialization; the requested symbol occurs once in the returned SM120 SASS'}
(out / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
          if p.is_file() and p.name != 'file_hashes.json'}
(out / 'file_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
print(json.dumps({'symbol_exact_match': True, 'static_hmma_occurrences': len(instructions), 'files': len(hashes)}))
