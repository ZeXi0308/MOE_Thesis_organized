"""Quality-only intervention on already gate-weighted expert outputs.

Reference operations intentionally favor transparency over speed. Never use
this module's wall time as a serving-system result. Mode II emulates logical
owner partials on one GPU, including local ones (not a native EP wire path).
"""
from dataclasses import dataclass
import torch
import codec_reference as ref


@dataclass(frozen=True)
class Policy:
    mode: str = 'I'
    precision: str = 'bf16'  # bf16, fp8, nvfp4, tiered
    head: int = 1
    ep_size: int = 2
    signal: str = 'gate'  # gate or partial_norm (mode II)


def owner_ids(ids, experts, ep_size):
    if not 0 < ep_size <= experts:
        raise ValueError('invalid expert/EP sizes')
    # Matches vLLM 0.26 linear placement: remainders go to first ranks.
    base, remainder = divmod(experts,ep_size)
    boundaries=torch.tensor([(r+1)*base+min(r+1,remainder) for r in range(ep_size-1)],device=ids.device)
    return torch.bucketize(ids,boundaries,right=True)


def qdq(x, precision):
    if precision=='bf16' or x.numel()==0:return x
    flat=x.reshape(-1,x.shape[-1])
    if precision=='fp8':
        q,s=ref.fp8_quant(flat);y=(q.float()*s[:,None]).to(x.dtype)
    elif precision=='nvfp4':
        q,s,g=ref.nvfp4_quant(flat);y=ref.nvfp4_dequant(q,s,g,dtype=x.dtype)
    else:raise ValueError(precision)
    return y.reshape_as(x)


def transform(weighted, weights, ids, experts, policy):
    if weighted.ndim!=3 or weights.shape!=weighted.shape[:2] or ids.shape!=weights.shape:
        raise ValueError('Expected weighted [token,k,d] and routing [token,k]')
    if policy.mode not in ['I','II'] or policy.signal not in ['gate','partial_norm']:
        raise ValueError('invalid mode/signal')
    if policy.mode=='I':
        if policy.signal!='gate':raise ValueError('Mode I uses gate ranking')
        vectors=weighted
        score=weights.float()
    else:
        owner=owner_ids(ids,experts,policy.ep_size)
        vectors=torch.zeros((weighted.shape[0],policy.ep_size,weighted.shape[2]),device=weighted.device,dtype=weighted.dtype)
        score=torch.zeros(weighted.shape[0],policy.ep_size,device=weighted.device)
        # FP32 local sum then model dtype, like sum along expert axis.
        for rank in range(policy.ep_size):
            mask=owner==rank
            vectors[:,rank]=(weighted.float()*mask[...,None]).sum(1).to(weighted.dtype)
            score[:,rank]=(weights.float()*mask).sum(1)
        if policy.signal=='partial_norm':score=vectors.float().norm(dim=-1)
    if policy.precision!='tiered':return qdq(vectors,policy.precision)
    if not 0<=policy.head<=vectors.shape[1]:raise ValueError('invalid head count')
    order=torch.argsort(score,dim=1,descending=True,stable=True)
    head=torch.zeros_like(score,dtype=torch.bool)
    head.scatter_(1,order[:,:policy.head],True)
    out=torch.empty_like(vectors)
    out[head]=qdq(vectors[head],'fp8')
    out[~head]=qdq(vectors[~head],'nvfp4')
    return out
