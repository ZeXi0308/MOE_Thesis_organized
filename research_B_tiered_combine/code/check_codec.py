"""Run reference semantics on CPU; --gpu also checks real Triton codec outputs."""
import argparse, json, pathlib
import torch
import codec_reference as ref


def main():
    p=argparse.ArgumentParser(); p.add_argument('--gpu',action='store_true'); p.add_argument('--output',required=True)
    a=p.parse_args(); device='cuda' if a.gpu else 'cpu'
    torch.manual_seed(1701)
    mid=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.],device=device)
    assert ref.e2m1_codes(mid).tolist()==[0,2,2,4,4,6,6]
    allcodes=torch.arange(16,dtype=torch.uint8,device=device)
    recovered=ref.e2m1_codes(ref.decode_e2m1(allcodes))
    expected=allcodes.clone(); expected[8]=0  # signed zero canonicalizes
    assert torch.equal(recovered,expected)
    results=[]
    if a.gpu:
        import codec
    for d in [16,48,512,2048,4096]:
        for kind in ['zero','random','outlier']:
            x=torch.randn(7,d,device=device,dtype=torch.bfloat16)
            if kind=='zero': x.zero_()
            if kind=='outlier': x[0,0]=1000; x[1]*=1e-6
            q,s,g=ref.nvfp4_quant(x)
            y=ref.nvfp4_dequant(q,s,g)
            assert torch.isfinite(y).all()
            if kind=='zero': assert torch.equal(x,y)
            row={'d':d,'kind':kind,'nvfp4_bytes':q.numel()+s.numel()+4,'fp8_bytes':x.numel()+4*x.shape[0]}
            if a.gpu:
                cq,cs,cg=codec.nvfp4_quant(x)
                assert torch.equal(g,cg),(d,kind,'global_scale')
                # Tolerate FP32 division-rounding differences only in scales;
                # payload must match to catch incorrect format/rounding.
                assert torch.equal(q,cq), (d,kind,'packed')
                assert torch.equal(s.view(torch.uint8),cs.view(torch.uint8)),(d,kind,'scale')
                assert torch.equal(y,codec.nvfp4_dequant(cq,cs,cg)),(d,kind,'dequant')
                fq,fs=ref.fp8_quant(x); aq,ass=codec.fp8_rowwise_quant(x)
                assert torch.equal(fq.view(torch.uint8),aq.view(torch.uint8)),(d,kind,'fp8')
                torch.testing.assert_close(fs,ass)
                torch.testing.assert_close(codec.fp8_rowwise_dequant(aq,ass),(fq.float()*fs[:,None]).to(x.dtype))
            results.append(row)
    scale_cases=[]
    if a.gpu:
        # Boundaries exercise partial blocks and the second-stage reduction,
        # including the largest benchmark tensor without costly full QDQ.
        for elements in [16,8176,8192,8208,100016,67108864]:
            x=torch.zeros((elements//16,16),device=device,dtype=torch.bfloat16)
            x[-1,-1]=-13.25
            expected=(x.float().abs().amax()/2688.).clamp_min(1e-12).reshape(1)
            actual=codec.nvfp4_tensor_scale(x)
            assert torch.equal(expected,actual),(elements,'scale reduction')
            scale_cases.append({'elements':elements,'scale':actual.item()})
    out={'status':'PASS','scope':'GPU Triton and CPU reference' if a.gpu else 'CPU reference only; GPU kernels unexecuted','cases':results,'scale_cases':scale_cases}
    pathlib.Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out))

if __name__=='__main__':main()
