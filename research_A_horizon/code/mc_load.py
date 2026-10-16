"""Extend C's synchronous MC target qualification to first-admission LOAD.

The original peak, old-owner guards, native fit, FIFO barrier, and accounting
are retained. A delayed new request consumes no more than the existing
prompt-plus-one-token-per-round envelope, which never credits its release.
This does not support recovery targets or already admitted LOAD owners.
"""
import ast
import inspect
import textwrap


def make_gate(mc):
    original=mc.Gate
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.before_allocate)))
    counts={'target':0,'super':0}
    class Extend(ast.NodeTransformer):
        def visit_If(self,node):
            if (len(node.body)==1 and isinstance(node.body[0],ast.Assign)
                    and isinstance(node.body[0].value,ast.Tuple)
                    and isinstance(node.body[0].value.elts[0],ast.Constant)
                    and node.body[0].value.elts[0].value=='not_new_local_prefill'):
                node.test=ast.BoolOp(op=ast.And(),values=[node.test,
                    ast.parse('not self.supported_load(req, kwargs)',mode='eval').body])
                counts['target']+=1
            return self.generic_visit(node)
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='super' and not node.args:
                counts['super']+=1
                return ast.parse('super(_original_gate, self)',mode='eval').body
            return self.generic_visit(node)
    tree=ast.fix_missing_locations(Extend().visit(tree))
    if counts!={'target':1,'super':1}:raise RuntimeError('MC source boundary changed')
    namespace=dict(original.before_allocate.__globals__,_original_gate=original)
    exec(compile(tree,__file__+'[C-before_allocate]','exec'),namespace)
    qualified=namespace['before_allocate']

    class LoadGate(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.mode='mc_budget_new_load'
            self.load_activity=dict(target_attempts=0,eligible=0,requested=0,successful=0)

        def supported_load(self,req,kwargs):
            external=kwargs.get('num_external_computed_tokens',0)
            manager=self.s.kv_cache_manager.coordinator.single_type_managers[0]
            return (kwargs.get('delay_cache_blocks') is True
                and kwargs['num_new_tokens']==0 and type(external) is int
                and 0<external<=req.num_prompt_tokens
                and req.status.name=='WAITING' and req.num_preemptions==0
                and req.num_computed_tokens==req.num_in_flight_tokens==req.num_output_tokens==0
                and req.num_tokens==req.num_prompt_tokens
                and not manager.req_to_blocks.get(req.request_id)
                and not kwargs.get('num_new_computed_tokens',0)
                and not kwargs.get('num_lookahead_tokens',0)
                and not kwargs.get('num_encoder_tokens',0))

        def before_allocate(self,req,*,token_budget,**kwargs):
            check=qualified(self,req,token_budget=token_budget,**kwargs)
            row=check['row']
            if row is not None:
                row['mc_new_load_target']=self.supported_load(req,kwargs)
                if row['mc_new_load_target'] and row['mc_attempted']:
                    self.load_activity['target_attempts']+=1
                    self.load_activity['eligible']+=row['mc_eligible'] is True
                    self.load_activity['requested']+=row['changed_by_mc_budget']
            return check

        def admitted(self,req):
            authorization=self.mc_authorizations.get(req.request_id)
            super().admitted(req)
            if authorization and authorization.get('mc_new_load_target'):
                self.load_activity['successful']+=1

        def report(self):
            result=super().report()
            result['new_load_adaptation']=dict(self.load_activity,
                semantics='New WAITING first-admission LOAD only. Full prompt charged at h=0; '
                'delayed target growth is bounded by the original earliest-growth envelope. '
                'All original old-owner guards, native fit, peak calculation, accounting and FIFO remain. '
                'This is a runtime adaptation of existing MC, not a new scheduling method.')
            return result
    return LoadGate


def install(mc,scheduler,cap,budget):
    # Reuse the original installer, including all additional source/runtime pins.
    namespace=dict(mc.install.__globals__,Gate=make_gate(mc))
    exec(compile(textwrap.dedent(inspect.getsource(mc.install)),
                 __file__+'[C-install]','exec'),namespace)
    return namespace['install'](scheduler,cap,budget,allow_scheduled_prefill=True)
