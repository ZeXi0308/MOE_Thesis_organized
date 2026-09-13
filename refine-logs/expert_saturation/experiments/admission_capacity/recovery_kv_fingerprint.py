"""Exact logical KV bytes for the pinned FLASH_ATTN (B,H,N,2D) layout.

Diagnostic only: CPU transfers and SHA hashing must stay outside performance
claims. Does not checkpoint/restore an engine or certify RNG/allocator state.
"""
import hashlib


def request_kv_digest(cache, block_ids, computed_tokens, *, block_size=16):
    import torch
    if cache.ndim != 4 or cache.shape[2] != block_size:
        raise ValueError('requires verified FLASH_ATTN logical (blocks,heads,tokens,2D) tensor')
    if computed_tokens < 0 or len(set(block_ids)) != len(block_ids):
        raise ValueError('invalid computed length or duplicate block ownership')
    required = (computed_tokens + block_size - 1) // block_size
    if len(block_ids) < required or any(i < 0 or i >= cache.shape[0] for i in block_ids):
        raise ValueError('insufficient or out-of-range blocks')
    digest = hashlib.sha256()
    remaining = computed_tokens
    byte_count = 0
    # Stream chunks: no full-cache copy or full request GPU concatenation.
    for start in range(0, required, 32):
        ids = torch.tensor(block_ids[start:min(start+32,required)],device=cache.device,dtype=torch.long)
        logical = cache.index_select(0,ids).permute(0,2,1,3).contiguous()
        logical = logical.reshape(-1,cache.shape[1],cache.shape[3])[:remaining]
        content = logical.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(content)
        byte_count += len(content)
        remaining -= logical.shape[0]
    if remaining:
        raise ValueError('incomplete KV byte stream')
    return dict(sha256=digest.hexdigest(),bytes=byte_count,tokens=computed_tokens,
                dtype=str(cache.dtype),heads=cache.shape[1],packed_head_size=cache.shape[3])


def selfcheck():
    import torch
    # Distinct valid logical values; physical placement must not affect digest.
    a=torch.arange(6*2*16*8,dtype=torch.float32).reshape(6,2,16,8)
    reference=request_kv_digest(a,[1,3],19)
    b=a.clone();b[4]=a[1];b[2]=a[3]
    assert request_kv_digest(b,[4,2],19)==reference
    # Uncomputed tail and free blocks must not enter the fingerprint.
    b[2,:,3:,:]+=10000;b[0]+=10000
    assert request_kv_digest(b,[4,2],19)==reference
    b[2,0,2,0]+=1
    assert request_kv_digest(b,[4,2],19)['sha256']!=reference['sha256']
    assert request_kv_digest(a,[],0)['bytes']==0
    try:request_kv_digest(a,[1],19)
    except ValueError:pass
    else:raise AssertionError('missing block accepted')
    assert reference['bytes']==19*2*8*4
    large=torch.arange(40*2*16*8,dtype=torch.float32).reshape(40,2,16,8).to(torch.bfloat16)
    streamed=request_kv_digest(large,list(range(34)),529)
    expected=large[:34].permute(0,2,1,3).contiguous().reshape(-1,2,8)[:529]
    assert streamed['sha256']==hashlib.sha256(expected.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert streamed['bytes']==529*2*8*2
    return dict(status='CPU_SELF_CHECK_PASS',cuda_initialized=torch.cuda.is_initialized(),
                layout='B,H,N,2D',checks=['physical relocation','unused tail exclusion',
                'valid-byte mutation','empty cache','missing block rejection','byte count','BF16 multi-chunk stream'],
                native_integration='UNRUN')


if __name__=='__main__':
    import json
    print(json.dumps(selfcheck(),indent=2))
