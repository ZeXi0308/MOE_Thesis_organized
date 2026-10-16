"""Gate-ranked contribution shares, not KL damage or causal importance.

Keep per-document/per-layer sums so document-level uncertainty remains possible.
The input is the already gate-weighted expert output before reduction.
"""
import torch


class RankProfile:
    def __init__(self):
        self.layers={}

    def __call__(self,name,weighted,weights,ids,experts):
        order=torch.argsort(weights.float(),dim=1,descending=True,stable=True)
        norms=weighted.float().norm(dim=-1).gather(1,order)
        gates=weights.float().gather(1,order)
        energy=norms.square()
        norm_total=norms.sum(1,keepdim=True)
        energy_total=energy.sum(1,keepdim=True)
        values=torch.stack((norms.sum(0),
            (norms/norm_total.clamp_min(1e-30)).sum(0),
            (energy/energy_total.clamp_min(1e-30)).sum(0),gates.sum(0)))
        if name not in self.layers:
            self.layers[name]={'sum':values,'tokens':0,'calls':0,
                               'zero_norm':torch.zeros((),device=weighted.device,dtype=torch.int64)}
        else:self.layers[name]['sum']+=values
        row=self.layers[name]
        row['tokens']+=weighted.shape[0];row['calls']+=1
        row['zero_norm']+=(norm_total==0).sum()

    def finish(self):
        result={}
        for name,row in self.layers.items():
            values=row['sum'].cpu().tolist()
            result[name]={'tokens':row['tokens'],'calls':row['calls'],
                'zero_norm_tokens':int(row['zero_norm'].item()),
                **dict(zip(('norm_sum_by_gate_rank','norm_share_sum_by_gate_rank',
                            'energy_share_sum_by_gate_rank','gate_sum_by_gate_rank'),values))}
        return result
