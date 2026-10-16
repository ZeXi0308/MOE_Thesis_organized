"""Worker-local full-distribution KL for matched eager, one-document forwards.

Every compute_logits call is included, including the final next-token
prediction; prompt NLL is collected independently by the vLLM API. Baseline
log-probabilities live on CPU while evaluating a single document; only metrics
need to be retained. This is quality instrumentation, not timing code.
"""
import types
import torch


class LogitsCapture:
    def __init__(self,model):
        self.model=model;self.original=model.compute_logits;self.reference=[]
        self.mode='off';self.index=0;self.token_kls=[];self.shapes=[]

    def install(self):
        def wrapped(model,*args,**kwargs):
            logits=self.original(*args,**kwargs)
            if self.mode!='off' and logits is not None:
                q=torch.log_softmax(logits.float(),dim=-1)
                self.shapes.append(list(q.shape))
                if self.mode=='baseline':self.reference.append(q.cpu())
                else:
                    if self.index>=len(self.reference):raise RuntimeError('extra compute_logits call')
                    p=self.reference[self.index].to(q.device)
                    if p.shape!=q.shape:raise RuntimeError('logit shape/order mismatch')
                    self.token_kls.extend(((p.exp()*(p-q)).sum(-1)).flatten().cpu().tolist())
                self.index+=1
            return logits
        self.model.compute_logits=types.MethodType(wrapped,self.model)
        return self

    def begin(self,baseline=False):
        if baseline:self.reference=[]
        elif not self.reference:raise RuntimeError('baseline required')
        self.mode='baseline' if baseline else 'candidate';self.index=0;self.token_kls=[];self.shapes=[]

    def finish(self):
        if self.index==0:raise RuntimeError('no logits captured')
        if self.mode=='candidate' and self.index!=len(self.reference):raise RuntimeError('missing compute_logits call')
        result={'mode':self.mode,'calls':self.index,'shapes':self.shapes,'token_kl':self.token_kls,
                'mean_kl':sum(self.token_kls)/len(self.token_kls) if self.token_kls else 0.}
        self.mode='off';return result

    def restore(self):
        self.model.compute_logits=self.original
