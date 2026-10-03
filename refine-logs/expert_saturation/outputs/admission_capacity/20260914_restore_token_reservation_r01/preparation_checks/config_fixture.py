#!/usr/bin/env python3
"""CPU configuration only: valid arm differences and one mismatched variant."""
from copy import deepcopy
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'experiments/admission_capacity'))
import analyze_restore_token_reservation as analyzer

preparation=HERE.parent/'preparation'
config=analyzer.read(preparation/'pkg/inputs_preparation/prepared/long/config.json')
specs=analyzer.read(preparation/'pkg/campaign.json')['cells'][:3]
meta=analyzer.read(preparation/'preparation.json')
engine=analyzer.read(Path(meta['reference_cell'])/'engine_args.json')
rows=[]
for spec in specs:
    cfg=dict(config,variant=spec['variant'],completion_policy='restore_token_reservation_'+spec['variant'],
        component=dict(boost=True,threshold=200,quantum=10,base_order='FCFS',
            backend='priority packing / current-history reservation / native recompute',
            packing=spec['packing'],complete_restores=spec['complete_restores']=='on',
            reserve_ready_tokens=spec['reserve_ready_tokens']=='on'))
    assert analyzer.spec_config(cfg,spec,config)
    rows.append(dict(config=cfg,engine=engine))
assert analyzer.pair_settings(rows[1],rows[2])
assert analyzer.pair_settings(rows[0],rows[2])
bad=deepcopy(rows[2]['config']);bad['variant']='guard_all'
assert not analyzer.spec_config(bad,specs[2],config)
out=dict(status='PASS',evidence_type='CPU_CONFIG_ONLY',runtime_executions=0,
    source='Constructed configurations from frozen input config and three campaign specs; no measurement rows.',
    valid_cell_configs=3,valid_cross_arm_shared_settings=['guard_all vs guard_residual','fit_scan vs guard_residual'],
    mismatched_variant_rejected=True,analyzer_sha256=analyzer.sha(Path(analyzer.__file__)),
    metadata_sha256=analyzer.sha(preparation/'preparation.json'))
path=HERE/'config_fixture.json';assert not path.exists(),'preserve fixture receipt'
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
