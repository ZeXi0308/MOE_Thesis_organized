"""Build the local raw-point recovery figure with installed TeX/Poppler only.

Usage: python3 figures/build_recovery_next_output.py
Reads completed episodes, checks fixed output indices, writes provenance/TeX,
compiles the PDF, and renders a PNG for visual review. No GPU packages/network.
"""
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NAME = 'recovery_next_output'
PANELS = [
    dict(name='E249', group='oct07_one_event_r01', target='measured/E249',
         target_index=76, peer='measured/E252', peer_index=48,
         target_token=3645, peer_token=15, xmax=260, ticks='0,50,100,150,200,250',
         title='(a) E249: R delays the peer', order='R/H/H/R'),
    dict(name='E223', group='oct07_e223_r01', target='measured/E223',
         target_index=541, peer='measured/E227', peer_index=422,
         target_token=14297, peer_token=281, xmax=1100, ticks='0,200,400,600,800,1000',
         title='(b) E223: waiting shifts to the peer', order='H/R/R/H'),
]


def read(path):
    return json.loads(path.read_text())


def collect():
    points, provenance = [], {}
    for panel in PANELS:
        group = ROOT / 'execution' / panel['group']
        summary = read(group / 'one_event_summary.json')
        assert summary['validation_passed'] and not summary['group_failures']
        repeats = {'host': 0, 'recompute': 0}
        cells = sorted(summary['full_service_cells'], key=lambda cell: cell['cell'])
        assert len(cells) == 4
        for cell in cells:
            path = group / cell['cell'] / 'raw.json'
            data = path.read_bytes()
            raw = json.loads(data)
            assert read(path.parent / 'status.json')['status'] == 'COMPLETE'
            commits = [c for c in raw['commits'] if c.get('target_committed')]
            assert len(commits) == 1
            commit = commits[0]
            assert commit['external_id'] == panel['target']
            action = commit['actual_action']
            assert action == cell['policy'] and action in repeats
            repeats[action] += 1
            requests = {r['external_id']: r for r in raw['requests']}
            for role in ['target', 'peer']:
                request = requests[panel[role]]
                index = panel[role + '_index']
                assert bisect_right(request['token_times_s'], commit['decision_s']) == index
                assert request['output_token_ids'][index] == panel[role + '_token']
                output_time = request['token_times_s'][index]
                points.append(dict(panel=panel['name'], cell=cell['cell'], action=action,
                    action_repetition=repeats[action], role=role,
                    external_id=panel[role], output_token_index=index,
                    output_token_id=request['output_token_ids'][index],
                    target_decision_s=commit['decision_s'], output_s=output_time,
                    delay_ms=1000 * (output_time - commit['decision_s'])))
            provenance[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        assert repeats == {'host': 2, 'recompute': 2}
    return dict(points=points, raw_sha256=provenance,
        definition='Milliseconds from the unique target commit decision_s to each specified request output; fixed indices are verified as the first strictly later output.',
        display='Individual runs only. Vertical offsets separate action/repetition; no aggregation, error bars, confidence intervals, or wall-time sum.',
        limitation='Local next-output latency includes execution, waiting and possible re-preemption; later output sequences differ across actions, so it is not an equal-work full-service or adaptive-policy gain.')


def source(data):
    lines = [r'''\documentclass[border=5pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc}
\pgfplotsset{compat=1.18}
\definecolor{hostblue}{HTML}{2166AC}
\definecolor{recomputered}{HTML}{B2182B}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
 group style={group size=2 by 1,horizontal sep=2.0cm},
 width=6.8cm,height=4.4cm,scale only axis,
 xmin=0,ymin=-0.34,ymax=1.34,
 ytick={0,1},ytick style={draw=none},
 xlabel={Time from target decision to next output (ms)},
 xlabel style={font=\fontsize{8}{9}\selectfont,yshift=-2pt},
 tick label style={font=\fontsize{8}{9}\selectfont},
 title style={font=\fontsize{9}{11}\selectfont,align=center},
 axis x line*=bottom,axis y line*=left,
 axis line style={black!55},xmajorgrids=true,ymajorgrids=true,
 grid style={black!12},tick align=outside,
 legend style={draw=none,font=\fontsize{8}{9}\selectfont,/tikz/every even column/.append style={column sep=12pt}},
 legend columns=2,
 every axis plot/.append style={only marks,mark size=2.6pt,line width=0.8pt},
]
''']
    styles = [('host', 1, 'hostblue', '*', 0.13),
              ('host', 2, 'hostblue', 'o', 0.043),
              ('recompute', 1, 'recomputered', 'square*', -0.043),
              ('recompute', 2, 'recomputered', 'square', -0.13)]
    for panel_index, panel in enumerate(PANELS):
        title = panel['title'] + r'\\[-1pt]{\footnotesize Run order: ' + panel['order'] + '}'
        legend = 'legend to name=recoverylegend,' if panel_index == 0 else ''
        lines.append(r'\nextgroupplot[' + legend + 'xmax=' + str(panel['xmax']) + ',xtick={' + panel['ticks'] +
                     '},yticklabels={Peer ' + panel['peer'].split('/')[-1] + ',Target ' +
                     panel['target'].split('/')[-1] + '},title={' + title + '}]\n')
        for action, repeat, color, mark, offset in styles:
            points = [p for p in data['points'] if p['panel'] == panel['name'] and
                      p['action'] == action and p['action_repetition'] == repeat]
            assert len(points) == 2
            coordinates = ' '.join(f"({p['delay_ms']:.9f},{(1 if p['role'] == 'target' else 0)+offset:.3f})" for p in points)
            forget = ',forget plot' if panel_index or repeat == 2 else ''
            lines.append(f'\\addplot[color={color},mark={mark}{forget}] coordinates {{{coordinates}}};\n')
            if not panel_index and repeat == 1:
                lines.append(r'\addlegendentry{' + ('Host recovery (H)' if action == 'host' else 'Native recompute (R)') + '}\n')
    lines.append(r'''\end{groupplot}
\node[anchor=north] at ($(group c1r1.south west)!0.5!(group c2r1.south east)+(0,-1.0cm)$)
 {\pgfplotslegendfromname{recoverylegend}};
\node[anchor=north,font=\fontsize{7.5}{9}\selectfont,align=center]
 at ($(group c1r1.south west)!0.5!(group c2r1.south east)+(0,-1.48cm)$)
 {Filled / open markers: first / second repetition of each action. Vertical offsets only separate points.\\
  Each point is one measured run; the two x-axis ranges differ.};
\end{tikzpicture}
\end{document}
''')
    return ''.join(lines)


def main():
    data = collect()
    (HERE / (NAME + '_points.json')).write_text(json.dumps(data, indent=2) + '\n')
    tex = HERE / (NAME + '.tex')
    tex.write_text(source(data))
    caption = ('Raw next-output latencies from the target decision for two repetitions per action at each frozen recovery event (E249: R/H/H/R; E223: H/R/R/H), including subsequent waiting and execution; later outputs differ across actions, so these local measurements do not establish equal-work full-service or adaptive-policy gains.')
    (HERE / (NAME + '_caption.txt')).write_text(caption + '\n')
    build = HERE / '.build_recovery_next_output'
    build.mkdir(exist_ok=True)
    command = ['/Library/TeX/texbin/pdflatex', '-interaction=nonstopmode', '-halt-on-error',
               '-output-directory', str(build), str(tex)]
    for _ in range(2):
        result = subprocess.run(command, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout[-6000:])
    pdf = HERE / (NAME + '.pdf')
    shutil.copy2(build / (NAME + '.pdf'), pdf)
    qa = HERE / 'qa'
    qa.mkdir(exist_ok=True)
    subprocess.run(['pdftoppm', '-singlefile', '-r', '170', '-png', str(pdf), str(qa / NAME)], check=True)
    print(json.dumps(dict(pdf=str(pdf), raw_points=len(data['points']),
                          render=str(qa / (NAME + '.png'))), indent=2))


if __name__ == '__main__':
    main()
