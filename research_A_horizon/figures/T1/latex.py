"""Typeset the full run inventory without pooling, filtering, or imputation."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def escaped(value):
    return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def number(value, scale=1, digits=3):
    if value is None:
        return '--'
    if isinstance(value, int) and scale == 1:
        return str(value)
    return f'{value*scale:.{digits}f}'


def main():
    data = json.loads((HERE/'data.json').read_text())
    rows = data['rows']
    lines = [r'\small', r'\setlength{\tabcolsep}{5pt}']
    def table(caption, columns, body, alignment=None):
        lines.extend([r'\begin{longtable}{'+(alignment or 'l'+'r'*(len(columns)-1))+'}',
                      r'\caption{'+caption+r'}\\', r'\toprule',
                      ' & '.join(columns)+r'\\\midrule\endfirsthead', r'\toprule',
                      ' & '.join(columns)+r'\\\midrule\endhead',
                      r'\bottomrule\endfoot'])
        lines.extend(' & '.join(row)+r'\\' for row in body)
        lines.extend([r'\end{longtable}', r'\clearpage'])
    identifiers = [f'R{i:02d}' for i in range(1, len(rows)+1)]
    index = []
    for rid, row in zip(identifiers, rows):
        policy = row['policy']
        if policy == 'horizon':
            policy += f" {row['ha']}/{row['hr']}"
        policy_tex=escaped(policy)+f"; cap{row['cap']}"
        switches=[]
        if row.get('lease_victim')==0:switches.append('no victim lease')
        if row.get('stall_order')==0:switches.append('no stall ordering')
        if row.get('estimate_hr')==1:switches.append('EWMA Hr')
        if switches:policy_tex+=r'\newline '+escaped(', '.join(switches))
        protocol = row['warmup']
        if row['comparison_use'] == 'download-overlap diagnostic':
            protocol += ', overlap'
        index.append([rid, escaped(row['session'])+r'\newline '+escaped(row['cell']),
            escaped(str(row['model_revision'])[:8]),
            escaped(f"{row['workload']} / {row['kv_gib']} / {row['arrival_rate']}"),
            policy_tex, escaped(row['code_snapshot']),
            escaped(protocol)+r'\newline '+escaped(row['status'])])
    table('Run inventory. Conditions are workload / KV GiB / requests per second. '
          'Overlap marks diagnostic download interference. Run IDs refer only to this generated inventory.',
          ['Run', 'Session and cell', 'Model SHA', 'Conditions', 'Policy', 'Code', 'Protocol/status'],
          index, r'lp{3.0in}lp{1.05in}p{1.35in}lp{1.1in}')
    groups = [
        ('Service metrics. Times in seconds; throughput in output tokens/s; joint goodput in requests/s.',
         [('TTFT P50','ttft_s_p50',1),('TTFT P95','ttft_s_p95',1),('Gap P99','maxgap_s_p99',1),
          (r'Stall (\%)','stall_rate',100),('Flow mean','flow_s_mean',1),('Flow P95','flow_s_p95',1),
          ('Throughput','output_token_s',1),('Joint GP','joint_goodput_rps',1)]),
        ('Request population and termination. Missing values are unavailable, not zero. '
         'Periodic output is a sanity check, not a quality-equivalence test.',
         [('Planned','planned',1),('Arrived','arrived',1),('Complete','completed',1),
          ('Failed','failed',1),('Unfinished','unfinished',1),('Not arrived','not_arrived',1),
          ('EOS','finish_stop',1),('Length cap','finish_length',1),
          (r'Periodic (\%)','periodic_last512_fraction',100)]),
        ('Mechanisms and costs. Preempted counts requests; events counts preemptions. '
         'Scheduler is whole-scheduler mean microseconds/step, not isolated adapter overhead.',
         [('Preempted','preempted_requests',1),('Events','preemption_count',1),
          ('Thrash','thrash_count',1),('MC relax','mc_relaxations',1),
          ('Scheduler','scheduler_us_per_step',1),('Control (s)','controller_seconds',1),
          ('Ledger (s)','ledger_seconds',1),('Window (s)','observation_s',1)]),
        ('Reservation samples. Percentages of KV blocks at KNOWN gate attempts; '
         'these are neither uniform time samples nor confidence intervals.',
         [(r'Committed P50 (\%)','peer_committed_ratio_p50',100),
          (r'Committed P95 (\%)','peer_committed_ratio_p95',100),
          (r'Growth P50 (\%)','peer_horizon_growth_ratio_p50',100),
          (r'Growth P95 (\%)','peer_horizon_growth_ratio_p95',100)])]
    for caption, specs in groups:
        table(caption, ['Run']+[s[0] for s in specs],
              [[rid]+[number(row[key],scale) for _,key,scale in specs]
               for rid,row in zip(identifiers,rows)])
    (HERE/'table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(runs=len(rows), tables=1+len(groups))))


if __name__ == '__main__':
    main()
