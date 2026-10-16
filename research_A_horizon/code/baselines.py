"""Unchanged C admission policies with HORIZON's common STORE and measurement hooks.

MC is the guarded synchronous scheduled-prefill adaptation, not C v25's
asynchronous engine. Vendor hashes identify the original, unmodified sources.
"""
from functools import partial
import hashlib
import json
import inspect
import textwrap
from pathlib import Path
import sys
import time

import horizon_sched as hz


class Baseline(hz.Adapter):
    def __init__(self,scheduler,cfg,*,policy,cap):
        super().__init__(scheduler,cfg)
        self.policy,self.cap=policy,cap
        self.remove_gate=None

    def attach(self,native):
        vendor=Path(__file__).with_name('vendor_c')
        manifest=json.loads((vendor/'sources.json').read_text())
        for name,row in manifest.items():
            if hashlib.sha256((vendor/name).read_bytes()).hexdigest()!=row['sha256']:
                raise RuntimeError('Modified C baseline: '+name)
        sys.path.insert(0,str(vendor))
        try:
            import declared_budget
            import mc_budget
        finally:
            sys.path.remove(str(vendor))
        for module in (declared_budget,mc_budget,declared_budget.native):
            if Path(module.__file__).resolve().parent!=vendor.resolve():
                raise RuntimeError('Another module owns C baseline imports')
        # Retain common lifecycle/STORE hooks, but use C's actual schedule patch.
        super().attach(native,schedule_hook=False)
        budget=self.m.block_pool.get_num_free_blocks()
        # Extend only the model-family guard; retain exact runtime/layout pins.
        original_qualify=declared_budget.native.qualify
        family=self.s.vllm_config.model_config.hf_config.model_type
        if family=='qwen3':
            hf=self.s.vllm_config.model_config.hf_config
            assert not hf.use_sliding_window and hf.sliding_window is None
            assert hf.num_hidden_layers==36 and hf.num_key_value_heads==8 and hf.head_dim==128
            source=textwrap.dedent(inspect.getsource(original_qualify))
            anchor="assert s.vllm_config.model_config.hf_config.model_type == 'olmoe'"
            assert source.count(anchor)==1
            source=source.replace(anchor,"assert s.vllm_config.model_config.hf_config.model_type == 'qwen3'")
            ns=dict(original_qualify.__globals__)
            exec(compile(source,'<Qwen-layout-qualified-C-baseline>','exec'),ns)
            declared_budget.native.qualify=ns['qualify']
            self.data['baseline_model_adaptation']='Qwen3-8B full attention; model guard extended, all source/layout/fit checks retained'
        try:
            if self.policy=='mc-load':
                from mc_load import install as install_load
                gate,self.remove_gate=install_load(mc_budget,self.s,self.cap,budget)
            elif self.policy=='mc':
                gate,self.remove_gate=mc_budget.install(self.s,self.cap,budget,
                                                       allow_scheduled_prefill=True)
            else:
                gate,self.remove_gate=declared_budget.install(self.s,self.cap,budget)
        finally:
            declared_budget.native.qualify=original_qualify
        admitted=gate.admitted
        def on_admitted(request):
            admitted(request)
            self.allocated(request,request.status.name=='WAITING_FOR_REMOTE_KVS')
        self._replace(gate,'admitted',on_admitted)
        patched=self.s.schedule
        def schedule(*args,**kwargs):
            start=time.perf_counter()
            self.begin()
            try:return patched(*args,**kwargs)
            finally:
                self.data['scheduler_seconds']+=time.perf_counter()-start
                self._count('schedule_calls')
        self._replace(self.s,'schedule',schedule)
        self.data.update(baseline_policy=self.policy,baseline_sources=manifest,
            complexity='unchanged C policy; measured whole scheduler wrapper',
            guarantee='C synchronous guarded component; not the asynchronous v25 system')
        return self

    def uninstall(self):
        if not self.active:return self.data
        result=super().uninstall()
        if self.remove_gate is not None:
            result['baseline']=self.remove_gate()
            self.remove_gate=None
        return result


def install_baseline(scheduler,policy,cap=256):
    if policy not in ('full-budget','mc','mc-load'):raise ValueError(policy)
    return hz.install(scheduler,hz.Config(0,0,False,False,False),
                      _adapter_factory=partial(Baseline,policy=policy,cap=cap))
