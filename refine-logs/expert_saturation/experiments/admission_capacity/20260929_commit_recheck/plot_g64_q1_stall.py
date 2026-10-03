"""Plot the observed output plateau, without imputing future tokens or completions."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent
cell = base / 'moe-a-g64-perf-remaining-session-r02-20260930/cell-00-ltr_t30_q1'
path = cell / 'archive/raw.json'
expected = json.loads((cell / 'output_sha256.json').read_text())['raw.json']
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
raw = json.loads(path.read_text())
assert raw['status'] == 'INCOMPLETE' and raw['error'] == 'runtime_limit'
assert len(raw['requests']) == 64
end = raw['observation_end_s']
tokens = sorted(t for r in raw['requests'] for t in r['token_times_s'])
complete = sorted(r['completion_s'] for r in raw['requests'] if r['status'] == 'completed')
out = base / 'g64_q1_stall_figure'
out.mkdir(exist_ok=False)
fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.6), sharex=True, constrained_layout=True)
axes[0].step([0, *tokens, end], [0, *range(1, len(tokens)+1), len(tokens)], where='post', color='#176b91', lw=1.8)
axes[0].set_ylabel('Returned tokens')
axes[0].set_title('G64 T30/Q1: observed implementation liveness failure', loc='left', fontsize=11)
axes[0].annotate(f'{end-tokens[-1]:.2f} s with no new output', xy=(100, len(tokens)), xytext=(53, 14500), arrowprops={'arrowstyle':'->','color':'#a33939'}, color='#a33939', fontsize=10)
axes[1].step([0, *complete, end], [0, *range(1, len(complete)+1), len(complete)], where='post', color='#176b91', lw=1.8)
axes[1].axhline(64, color='#777777', linestyle='--', lw=1, label='64 arrived requests')
axes[1].set_ylim(-1, 69)
axes[1].set_ylabel('Completed requests')
axes[1].set_xlabel('Seconds from measurement start')
axes[1].legend(loc='center right', frameon=False, fontsize=9)
for ax in axes:
    ax.axvspan(tokens[-1], end, color='#b94c4c', alpha=.08)
    ax.set_xlim(0, end)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
fig.savefig(out/'observed_output_plateau.png', dpi=180)
fig.savefig(out/'observed_output_plateau.svg')
(out/'source.json').write_text(json.dumps({'raw_sha256':expected,'requests':64,'completed':len(complete),'unfinished':58,'last_output_s':tokens[-1],'observation_end_s':end,'scope':'Observed frozen implementation, not a conclusion about the Q=1 algorithm or unseen outputs'}, indent=2)+'\n')
print(out)
