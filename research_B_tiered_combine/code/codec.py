"""GPU-resident codecs; no host staging. NVFP4 packs even element in low nibble.

Finite, contiguous 2D inputs, d divisible by 16. FP32 row scales for FP8;
E4M3 block-16 scales plus one FP32 tensor scale for NVFP4. Dynamic tensor-scale
reduction is separate and included by nvfp4_quant; supplying a GPU scale makes
packing one launch. APIs allocate outputs; benchmark must disclose this.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _fp8q(X,Q,S,D:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0); i=tl.arange(0,B)
    x=tl.load(X+r*D+i,i<D,other=0).to(tl.float32)
    s=tl.maximum(tl.max(tl.abs(x),0)/448.,1e-12)
    tl.store(S+r,s)
    tl.store(Q+r*D+i,tl.minimum(tl.maximum(tl.div_rn(x,s),-448.),448.).to(tl.float8e4nv),i<D)


@tr.jit
def _fp8dq(Q,S,Y,D:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0); i=tl.arange(0,B)
    q=tl.load(Q+r*D+i,i<D,other=0.).to(tl.float32)
    tl.store(Y+r*D+i,q*tl.load(S+r),i<D)


@tr.jit
def _codes(x):
    a=tl.abs(x)
    c=(a>.25).to(tl.int32)+(a>=.75).to(tl.int32)+(a>1.25).to(tl.int32)+(a>=1.75).to(tl.int32)+(a>2.5).to(tl.int32)+(a>=3.5).to(tl.int32)+(a>5.).to(tl.int32)
    return c | ((x<0).to(tl.int32)<<3)


@tr.jit
def _nvq(X,Q,S,G,D:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0); i=tl.arange(0,B)
    x=tl.load(X+r*D+i,i<D,other=0).to(tl.float32)
    g=tl.load(G)
    blocks=tl.reshape(x,(B//16,16))
    raw=tl.div_rn(tl.max(tl.abs(blocks),1),6.*g)
    sf=tl.minimum(raw,448.).to(tl.float8e4nv)
    bi=tl.arange(0,B//16)
    tl.store(S+r*(D//16)+bi,sf,bi<D//16)
    scale=sf.to(tl.float32)*g
    z=tl.where(scale[:,None]>0,tl.div_rn(blocks,tl.maximum(scale[:,None],tl.full((),1e-38,tl.float32))),0.)
    codes=tl.reshape(_codes(z),(B//2,2))
    lo,hi=tl.split(codes)
    pi=tl.arange(0,B//2)
    tl.store(Q+r*(D//2)+pi,(lo|(hi<<4)).to(tl.uint8),pi<D//2)


@tr.jit
def _nvdq(Q,S,G,Y,D:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0); i=tl.arange(0,B)
    p=tl.load(Q+r*(D//2)+i//2,i<D,other=0).to(tl.int32)
    c=(p>>((i%2)*4))&15; m=c&7
    v=tl.where(m==0,0.,tl.where(m==1,.5,tl.where(m==2,1.,tl.where(m==3,1.5,tl.where(m==4,2.,tl.where(m==5,3.,tl.where(m==6,4.,6.)))))))
    s=tl.load(S+r*(D//16)+i//16,i<D,other=0.).to(tl.float32)
    y=(v*tl.where((c&8)!=0,-1.,1.))*(s*tl.load(G))
    tl.store(Y+r*D+i,y,i<D)


def _validate(x):
    if not x.is_cuda or x.ndim!=2 or not x.is_contiguous() or x.shape[0]<1 or x.shape[1]%16 or x.shape[1]<16:
        raise ValueError('Expected nonempty contiguous CUDA [rows,d], d divisible by 16')
    if x.dtype not in (torch.bfloat16,torch.float16,torch.float32):
        raise ValueError('Expected bf16/fp16/fp32')


@tr.jit
def _absmax_partial(X,P,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+i,i<N,other=0.).to(tl.float32)
    tl.store(P+tl.program_id(0),tl.max(tl.abs(x),0))


@tr.jit
def _absmax_scale(P,G,N:tl.constexpr,B:tl.constexpr):
    i=tl.arange(0,B)
    maximum=tl.max(tl.load(P+i,i<N,other=0.),0)
    # PyTorch's division by a scalar uses its FP32 reciprocal on CUDA.
    tl.store(G,tl.maximum(maximum*(1.0/2688.0),1e-12))


def nvfp4_tensor_scale(x):
    """Two reductions; avoid materializing float()/abs() over the full tensor."""
    _validate(x)
    chunks=tr.cdiv(x.numel(),8192)
    partial=torch.empty(chunks,device=x.device,dtype=torch.float32)
    global_scale=torch.empty(1,device=x.device,dtype=torch.float32)
    _absmax_partial[(chunks,)](x,partial,x.numel(),8192)
    _absmax_scale[(1,)](partial,global_scale,chunks,tr.next_power_of_2(chunks))
    return global_scale


def fp8_rowwise_quant(x):
    _validate(x)
    q=torch.empty_like(x,dtype=torch.float8_e4m3fn)
    s=torch.empty(x.shape[0],device=x.device,dtype=torch.float32)
    _fp8q[(x.shape[0],)](x,q,s,x.shape[1],tr.next_power_of_2(x.shape[1]))
    return q,s


def fp8_rowwise_dequant(q,s,dtype=torch.bfloat16):
    y=torch.empty(q.shape,device=q.device,dtype=dtype)
    _fp8dq[(q.shape[0],)](q,s,y,q.shape[1],tr.next_power_of_2(q.shape[1]))
    return y


def nvfp4_quant(x,global_scale=None):
    _validate(x)
    g=nvfp4_tensor_scale(x) if global_scale is None else global_scale
    if g.device!=x.device or g.dtype!=torch.float32 or g.numel()!=1:
        raise ValueError('global_scale must be one positive finite FP32 value on the input GPU')
    q=torch.empty((x.shape[0],x.shape[1]//2),device=x.device,dtype=torch.uint8)
    s=torch.empty((x.shape[0],x.shape[1]//16),device=x.device,dtype=torch.float8_e4m3fn)
    _nvq[(x.shape[0],)](x,q,s,g,x.shape[1],tr.next_power_of_2(x.shape[1]))
    return q,s,g


def nvfp4_dequant(q,s,g,dtype=torch.bfloat16):
    d=q.shape[1]*2
    y=torch.empty((q.shape[0],d),device=q.device,dtype=dtype)
    _nvdq[(q.shape[0],)](q,s,g,y,d,tr.next_power_of_2(d))
    return y
