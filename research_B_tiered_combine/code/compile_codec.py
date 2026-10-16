"""Compile all codec kernels to SM120 without creating a CUDA context."""
import argparse,json,pathlib
import triton,codec
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
cases=[(codec._fp8q,{'X':'*bf16','Q':'*fp8e4nv','S':'*fp32'}),
       (codec._fp8dq,{'Q':'*fp8e4nv','S':'*fp32','Y':'*bf16'}),
       (codec._nvq,{'X':'*bf16','Q':'*u8','S':'*fp8e4nv','G':'*fp32'}),
       (codec._nvdq,{'Q':'*u8','S':'*fp8e4nv','G':'*fp32','Y':'*bf16'})]
r=[]
for d in [48,2048,4096]:
    for fn,sig in cases:
        c=triton.compile(ASTSource(fn,signature=sig,constexprs={'D':d,'B':triton.next_power_of_2(d)}),target=GPUTarget('cuda',120,32))
        r.append({'kernel':fn.__name__,'d':d,'ptx_bytes':len(c.asm['ptx']),'cubin_bytes':len(c.asm['cubin'])})
        print(r[-1],flush=True)
for n in [1,64,8192]:
    for fn,sig,const in [
        (codec._absmax_partial,{'X':'*bf16','P':'*fp32'},{'N':n*8192,'B':8192}),
        (codec._absmax_scale,{'P':'*fp32','G':'*fp32'},{'N':n,'B':triton.next_power_of_2(n)})]:
        c=triton.compile(ASTSource(fn,signature=sig,constexprs=const),target=GPUTarget('cuda',120,32))
        r.append({'kernel':fn.__name__,'partials':n,'ptx_bytes':len(c.asm['ptx']),'cubin_bytes':len(c.asm['cubin'])})
        print(r[-1],flush=True)
pathlib.Path(a.output).write_text(json.dumps({'status':'PASS','scope':'offline SM120 compilation only; no GPU execution','cases':r},indent=2))
