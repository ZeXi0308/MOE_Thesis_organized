"""CPU-only token windows and arm definitions, shared by runner and checks."""
import math


def token_windows(tokens,context):
    """One-token overlap scores every document token after the first exactly once.

    Context resets at each block: this is block-context PPL, not full-history PPL.
    """
    if context<2:raise ValueError('context must be >=2')
    start=0
    while start<len(tokens)-1:
        end=min(start+context,len(tokens))
        yield {'start':start,'end':end,'token_ids':tokens[start:end]}
        start=end-1


def default_arms(top_k,ep_size=2):
    if top_k<2:raise ValueError('top_k must be >=2')
    arms=[{'name':'native','baseline':'native','policy':None},
          {'name':'I_bf16','baseline':'native','policy':{'mode':'I','precision':'bf16'},'profile':True}]
    for mode,units in [('I',top_k),('II',ep_size)]:
        base='native' if mode=='I' else 'II_bf16'
        if mode=='II':arms.append({'name':base,'baseline':base,'policy':{'mode':mode,'precision':'bf16','ep_size':ep_size}})
        for precision in ['fp8','nvfp4']:
            arms.append({'name':f'{mode}_{precision}','baseline':base,
                'policy':{'mode':mode,'precision':precision,'ep_size':ep_size}})
        for head in range(1,units):
            arms.append({'name':f'{mode}_tiered_h{head}','baseline':base,
                'policy':{'mode':mode,'precision':'tiered','head':head,'ep_size':ep_size}})
        if mode=='II':
            arms.append({'name':'II_tiered_norm_h1','baseline':base,
                'policy':{'mode':mode,'precision':'tiered','head':1,'signal':'partial_norm','ep_size':ep_size}})
    return arms


def validate_arms(arms):
    names=set();active=None
    for arm in arms:
        name=arm['name']
        if not name or name in names:raise ValueError('duplicate/empty arm name')
        names.add(name)
        if arm['baseline']==name:active=name
        elif arm['baseline']!=active:raise ValueError(f'{name}: baseline must precede its group')


def payload_bytes(policy,rows,width,top_k):
    """Logical payload incl. scales, excluding row permutation/control metadata."""
    if policy is None:policy={'mode':'I','precision':'bf16'}
    units=top_k if policy['mode']=='I' else policy.get('ep_size',2)
    total=rows*units;precision=policy['precision']
    if precision=='bf16':return 2*total*width
    if precision=='fp8':return total*(width+4)
    if precision=='nvfp4':return total*(width//2+width//16)+4
    if precision!='tiered':raise ValueError(precision)
    head=rows*policy['head'];tail=total-head
    return head*(width+4)+tail*(width//2+width//16)+(4 if tail else 0)


def ppl(nll,tokens):
    return math.exp(nll/tokens) if tokens else None
