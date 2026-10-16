"""Keep API-event and graph-amortized timing distinct in a compact CSV."""
import argparse,csv,json,pathlib
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args()
r=json.loads(pathlib.Path(a.input).read_text())
if r['status']!='COMPLETE':raise ValueError('Incomplete benchmark')
rows=[]
for x in r['results']:
    rows.append({'rows':x['rows'],'hidden':x['d'],'operation':x['operation'],
                 'eager_api_event_median_us':x['median_us'],
                 'eager_min_us':min(x['us_repeats']),'eager_max_us':max(x['us_repeats']),
                 'graph_amortized_median_us':x.get('graph_median_us',''),
                 'source':a.input})
with open(a.output,'w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
