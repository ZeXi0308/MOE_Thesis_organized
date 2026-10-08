#!/usr/bin/env python3
"""Fixed 1/16/32-page shadow coverage at actual first baseline allocations.

Usage: python3 headroom_coverage.py PREOUTPUT_COVERAGE.json [...] --output NEW.json
Reuses the source coverage alignment and labels; does not replay a controller.
"""
import argparse
import hashlib
import json
from pathlib import Path


EXTRA_PAGES = (1, 16, 32)


def analyze_cell(cell, source_path):
    path = Path(cell['cell'])
    alignment = cell['population_allocation_alignment']
    if alignment['unknown'] or alignment['matched'] != cell['planned_requests']:
        raise ValueError('Requires the existing complete first-allocation alignment: ' + str(path))
    if cell['unclassified_successful_preemptions']:
        raise ValueError('Pre-output victim labels include unresolved host ordering: ' + str(path))
    data = {}
    for name in ('raw', 'admission'):
        payload = (path / (name+'.json')).read_bytes()
        if hashlib.sha256(payload).hexdigest() != cell[name+'_sha256']:
            raise ValueError('Source differs from prior alignment: ' + str(path / (name+'.json')))
        data[name] = json.loads(payload)
    raw, admission = data['raw'], data['admission']
    if admission['probe_enabled'] is not False or admission['reservation_probe']['probe_enabled'] is not False:
        raise ValueError('Only nonintervened reservation baselines are supported')
    identity = {r[k]: r['request_id'] for r in raw['requests']
        for k in ('request_id', 'internal_request_id', 'external_request_id') if r.get(k)}
    population = {r['request_id'] for r in raw['requests']}
    first = {}
    for row in admission['decisions']:
        if row.get('native_allocation_result') is True:
            rid = identity[row['request_id']]
            if rid in first:
                raise ValueError('More than one successful new hook for an already aligned request: ' + rid)
            first[rid] = row
    if first.keys() != population or len(first) != cell['planned_requests']:
        raise ValueError('Successful first-allocation population differs from prior alignment')
    victims = {r['request_id'] for r in cell['requests']}
    if not victims <= population or len(victims) != cell['preoutput_after_firstprefill_requests']:
        raise ValueError('Victim IDs/count differ from source population')
    margins, r_full, unknown = {}, {}, {}
    for rid, row in first.items():
        if row.get('native_reserved_blocks') != 0:
            unknown[rid] = 'native_reserved_blocks is missing or nonzero; no repeated subtraction'
            continue
        keys = ('free_blocks', 'full_required_blocks', 'native_watermark_blocks')
        if any(type(row.get(k)) is not int or row[k] < 0 for k in keys):
            unknown[rid] = 'Missing or invalid current full-fit geometry'
            continue
        margin = row['free_blocks']-row['full_required_blocks']-row['native_watermark_blocks']
        if margin < 0:
            raise ValueError('Negative full-fit margin on successful first allocation: ' + rid)
        margins[rid] = margin
        r_full[rid] = row.get('full_commit_opportunity')
        if type(r_full[rid]) is not bool:
            raise ValueError('Missing R-full shadow flag on supported observation row: ' + rid)

    def coverage(selected):
        known = set(margins)
        covered, other, missed = selected & victims, selected-victims, (victims & known)-selected
        return dict(selected=len(selected), covered_preoutput_victim=len(covered), selected_other=len(other),
            missed_victim=len(missed), unknown_requests=len(unknown), unknown_victims=len(victims-known),
            selected_request_ids=sorted(selected), covered_victim_ids=sorted(covered),
            missed_victim_ids=sorted(missed), unknown_victim_ids=sorted(victims-known))

    return dict(cell=cell['cell'], source_preoutput_coverage=str(source_path.resolve()),
        raw_sha256=cell['raw_sha256'], admission_sha256=cell['admission_sha256'],
        planned_requests=len(population), preoutput_victim_requests=len(victims),
        first_allocation_alignment_reused=True, margin_observed_requests=len(margins),
        unknown_first_allocations=unknown,
        fixed_extra_headroom=[dict(extra_pages=extra,
            **coverage({rid for rid, margin in margins.items() if margin < extra})) for extra in EXTRA_PAGES],
        r_full_shadow=coverage({rid for rid, value in r_full.items() if value}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('coverage', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing output.')
    cells, seen = [], set()
    for path in args.coverage:
        source = json.loads(path.read_text())
        for cell in source['cells']:
            key = str(Path(cell['cell']).resolve())
            if key in seen:
                parser.error('Duplicate baseline cell: ' + key)
            seen.add(key)
            cells.append(analyze_cell(cell, path))
    result = dict(schema_version=1, independent_unit='run', fixed_extra_pages=list(EXTRA_PAGES),
        decision_rule='At each actual first successful new allocation, margin = free_blocks - '
            'full_required_blocks - native_watermark_blocks. For native_reserved_blocks==0, '
            'the fixed shadow selects iff margin < extra_pages. Nonzero/missing native reserve is unknown.',
        reference='R-full uses the recorded full_commit_opportunity at exactly the same first-successful '
            'allocation rows and population; no reservation-event target substitution.',
        inference='Only retrospective coverage of fixed, predeclared shadow rules. Victim labels come from '
            'later observed host preemption/token ordering and are never online features. Selected other '
            'means outside this observed pre-output-preemption label, not a harmful action or proven false '
            'positive. These counts do not establish executability after intervention, timing benefit, '
            'causal savings, or a reason to select a threshold after inspecting the results.', cells=cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print('run / baseline / rule: selected, covered victim, selected other, missed victim, unknown')
    for c in cells:
        path = Path(c['cell'])
        for label, row in [(str(r['extra_pages']), r) for r in c['fixed_extra_headroom']]+[('R-full', c['r_full_shadow'])]:
            print(path.parent.name, path.name, label, *(row[k] for k in
                ('selected', 'covered_preoutput_victim', 'selected_other', 'missed_victim', 'unknown_requests')))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
