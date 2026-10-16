"""Five post-warmup CUDA-event repeats per shape, no host transfers in codec.

Caller must own the shared GPU lock and verify exclusive GPU access. Outputs
include allocation in API calls and separate NVFP4 pack-only and dynamic-scale
quant timings. GB/s is decoded BF16 payload bytes / duration, not DRAM traffic.
"""
import argparse,json,pathlib,statistics,time
import torch
import codec


def measure(fn):
    for _ in range(10):fn()
    torch.cuda.synchronize()
    values=[]
    for _ in range(5):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record(); result=fn(); end.record(); end.synchronize()
        values.append(start.elapsed_time(end)*1000)
    return values


def measure_graph(fn, repeats=20):
    # Keep all repeated invocations on one stream in the captured graph. Divide
    # the complete event span by repetitions to amortize Python submission gaps.
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10): fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        for _ in range(repeats): result=fn()
    graph.replay();torch.cuda.synchronize()
    values=[]
    for _ in range(5):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();graph.replay();end.record();end.synchronize()
        values.append(start.elapsed_time(end)*1000/repeats)
    return values


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=pathlib.Path(a.output)
    if out.exists():raise FileExistsError(out)
    rows=[];torch.manual_seed(1701)
    for n in [256,1024,4096,16384]:
        for d in [2048,4096]:
            x=torch.randn(n,d,device='cuda',dtype=torch.bfloat16)
            fq,fs=codec.fp8_rowwise_quant(x);nq,ns,ng=codec.nvfp4_quant(x)
            tasks={'fp8_quant':lambda:codec.fp8_rowwise_quant(x),
                   'fp8_dequant':lambda:codec.fp8_rowwise_dequant(fq,fs),
                   'nvfp4_quant_dynamic_scale':lambda:codec.nvfp4_quant(x),
                   'nvfp4_pack_given_scale':lambda:codec.nvfp4_quant(x,ng),
                   'nvfp4_dequant':lambda:codec.nvfp4_dequant(nq,ns,ng)}
            for name,fn in tasks.items():
                us=measure(fn);med=statistics.median(us);graph_us=measure_graph(fn)
                rows.append({'rows':n,'d':d,'operation':name,'us_repeats':us,'median_us':med,'timing_scope':'eager API event span includes host submission gaps; graph repeats amortize them','graph_repetitions':20,'graph_us_repeats':graph_us,'graph_median_us':statistics.median(graph_us),'decoded_payload_GBps':2*n*d/med/1000})
            out.write_text(json.dumps({'status':'RUNNING','results':rows},indent=2))
    out.write_text(json.dumps({'status':'COMPLETE','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'results':rows},indent=2))

if __name__=='__main__':main()
