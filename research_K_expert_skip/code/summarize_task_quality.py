#!/usr/bin/env python3
"""Summarize actual run_task_quality suites without running models.

  python code/summarize_task_quality.py --suite runs/RUN/quality/suite.json \
      --output-dir figures/task_quality_RUN

Accept multiple suite.json files (or their containing directories). Remote job
paths are resolved first beside the supplied suite. Completed entries in a
partial suite remain usable; absent/unaccepted arms stay NA. No runs, formats,
task filters or seeds are pooled. MC results are prefill identity controls;
only GSM8K and free generation can exercise the decode policy. Harness sample
stderr is retained in the leaf CSV, never used as independent-run uncertainty.
Free windows retain source-article groups; repetition is not correctness.
GSM8K additionally exports document correctness and native/candidate discordance,
joined by official doc_id plus document/prompt/target hashes, separately per filter.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
MC = {'mmlu', 'hellaswag', 'arc_challenge'}


def compact(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def numeric(value):
    return float(value) if isinstance(value, (float, int, bool)) and math.isfinite(value) else None


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


class Reader:
    def __init__(self):
        self.sources = {}; self.notes = []

    def text(self, path):
        path = Path(path).resolve(); raw = path.read_bytes()
        self.sources[str(path)] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
        return raw.decode('utf-8')

    def json(self, path, default=None):
        return json.loads(self.text(path)) if Path(path).is_file() else default

    def rows(self, path):
        if not Path(path).is_file():
            return None
        text = self.text(path); lines = text.splitlines(); rows = []
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                if index == len(lines) and not text.endswith('\n'):
                    self.notes.append(f'Ignored an unfinished final JSONL record: {path}')
                else:
                    raise
        return rows


def job_directory(suite_path, job):
    recorded = Path(job['output']) if job.get('output') else None
    if Path(job['id']).name != job['id'] or (recorded and recorded.name != job['id']):
        raise ValueError(f"job id/output basename mismatch: {job['id']} {recorded}")
    candidates = [suite_path.parent/job['id']]
    if recorded is not None:
        candidates.append(recorded)
    return next((p for p in candidates if p.is_dir()), candidates[0])


def base_fields(suite_path, suite, job, config):
    args = {**suite.get('args', {}), **config.get('args', {})}
    return {'suite': str(suite_path), 'suite_status': suite.get('status', 'UNKNOWN'),
        'job': job['id'], 'job_status': job.get('status', 'UNKNOWN'), 'format': job.get('format'),
        'planned_configurations': len(job['arms']),
        'model': args.get('model'), 'seed': args.get('seed'), 'unit': job['unit'],
        'limit_per_leaf': args.get('limit') if job['unit'] != 'free_generation' else None,
        'max_tokens': args.get('max_tokens'), 'batch': args.get('batch'),
        'scope': 'MC_PREFILL_IDENTITY_CONTROL' if job['unit'] in MC else 'DECODE_GENERATION_FIRST_TOKEN_PREFILL'}


def sample_key(task, row):
    # Filter is deliberately absent: two extraction filters are one document.
    return task, row.get('doc_id'), row.get('doc_hash'), row.get('prompt_hash'), row.get('target_hash')


def sample_metrics(result):
    values = defaultdict(list); docs = defaultdict(set)
    for task, rows in result.get('samples', {}).items():
        for row in rows:
            docs[task].add(sample_key(task, row))
            for metric in row.get('metrics', []):
                value = numeric(row.get(metric))
                if value is not None:
                    values[f"{metric},{row.get('filter', 'none')}"].append(value)
    return {key: {'mean': mean(vals), 'sample_rows': len(vals)} for key, vals in values.items()}, docs


def gsm8k_pairs(common, job, loaded):
    """Use complete, unique raw populations; never pair a shared subset silently."""
    filters = ('strict-match', 'flexible-extract')
    documents, paired_documents, summaries, groups = [], [], [], {}
    for arm in job['arms']:
        if arm not in loaded:
            continue
        _, result, result_path, _, _ = loaded[arm]
        effective = result.get('n-samples', {}).get('gsm8k', {}).get('effective')
        for extraction in filters:
            rows = [(index, row) for index, row in enumerate(result.get('samples', {}).get('gsm8k', []))
                    if row.get('filter') == extraction]
            indexed, errors = {}, []
            for index, row in rows:
                doc_id = row.get('doc_id')
                identity = sample_key('gsm8k', row)
                correct = numeric(row.get('exact_match'))
                problems = []
                if not isinstance(doc_id, int) or isinstance(doc_id, bool) or doc_id < 0:
                    problems.append('missing_or_invalid_official_doc_id')
                if any(not isinstance(value, str) or not value for value in identity[2:]):
                    problems.append('missing_document_prompt_or_target_hash')
                if correct not in (0., 1.) or 'exact_match' not in row.get('metrics', []):
                    problems.append('missing_or_nonbinary_exact_match')
                key = compact(doc_id)
                if key in indexed:
                    problems.append('duplicate_doc_id_within_filter')
                errors.extend(problems)
                item = {**common, 'arm': arm, 'task': 'gsm8k', 'filter': extraction,
                    'doc_id': doc_id, 'doc_hash': row.get('doc_hash'),
                    'prompt_hash': row.get('prompt_hash'), 'target_hash': row.get('target_hash'),
                    'correct': int(correct) if correct in (0., 1.) else None,
                    'document_status': ';'.join(problems) if problems else 'OBSERVED',
                    'result_file': str(result_path), 'sample_index': index}
                documents.append(item)
                indexed[key] = item
            if not isinstance(effective, int) or isinstance(effective, bool) or effective < 1:
                errors.append('missing_or_invalid_harness_effective_population')
            elif len(rows) != effective:
                errors.append('raw_population_differs_from_harness_effective_population')
            groups[arm, extraction] = {'documents': indexed, 'errors': sorted(set(errors)),
                                      'raw_rows': len(rows), 'effective': effective}
    for arm in job['arms']:
        if arm == 'native':
            continue
        for extraction in filters:
            native, current = groups.get(('native', extraction)), groups.get((arm, extraction))
            reasons = []
            for label, group in (('native', native), ('candidate', current)):
                if group is None:
                    reasons.append(label + ':missing_accepted_raw_result')
                else:
                    reasons.extend(label + ':' + error for error in group['errors'])
            old_docs = native['documents'] if native else {}
            new_docs = current['documents'] if current else {}
            old_ids, new_ids = set(old_docs), set(new_docs)
            mismatch = [key for key in old_ids & new_ids
                        if any(old_docs[key][field] != new_docs[key][field]
                               for field in ('doc_hash', 'prompt_hash', 'target_hash'))]
            if old_ids != new_ids:
                reasons.append('different_official_doc_id_population')
            if mismatch:
                reasons.append('document_prompt_or_target_identity_mismatch')
            valid = not reasons
            row = {**common, 'arm': arm, 'task': 'gsm8k', 'filter': extraction,
                'pairing_status': 'PAIRED_COMPLETE_POPULATION' if valid else 'NOT_PAIRED',
                'pairing_reason': ';'.join(reasons) if reasons else 'same unique official doc_ids and document/prompt/target hashes',
                'native_raw_rows': native['raw_rows'] if native else None,
                'candidate_raw_rows': current['raw_rows'] if current else None,
                'native_effective_documents': native['effective'] if native else None,
                'candidate_effective_documents': current['effective'] if current else None,
                'native_ids_missing_from_candidate': len(old_ids-new_ids),
                'candidate_ids_missing_from_native': len(new_ids-old_ids),
                'identity_mismatches': len(mismatch),
                'paired_documents': len(old_docs) if valid else 0,
                **{key: None for key in ('native_only_correct', 'candidate_only_correct', 'both_correct',
                    'both_wrong', 'native_correct', 'candidate_correct', 'native_accuracy',
                    'candidate_accuracy', 'paired_accuracy_difference', 'paired_accuracy_difference_pp')}}
            if valid:
                counts = Counter()
                for key in sorted(old_ids, key=lambda value: old_docs[value]['doc_id']):
                    old, new = old_docs[key], new_docs[key]
                    before, after = old['correct'], new['correct']
                    category = ('both_correct' if before else 'candidate_only_correct') if after else ('native_only_correct' if before else 'both_wrong')
                    counts[category] += 1
                    paired_documents.append({**new, 'native_correct': before, 'candidate_correct': after,
                        'correctness_difference': after-before, 'discordance': category,
                        'native_result_file': old['result_file'], 'native_sample_index': old['sample_index']})
                count = len(old_docs)
                native_correct = counts['both_correct']+counts['native_only_correct']
                candidate_correct = counts['both_correct']+counts['candidate_only_correct']
                difference = (candidate_correct-native_correct)/count
                row.update({key: counts[key] for key in ('native_only_correct', 'candidate_only_correct', 'both_correct', 'both_wrong')})
                row.update(native_correct=native_correct, candidate_correct=candidate_correct,
                           native_accuracy=native_correct/count, candidate_accuracy=candidate_correct/count,
                           paired_accuracy_difference=difference, paired_accuracy_difference_pp=100*difference)
            summaries.append(row)
    return documents, paired_documents, summaries, groups


def termination(rows):
    if rows is None:
        return {key: None for key in ('observed_outputs', 'eos', 'stop', 'limit', 'fail', 'first_prediction_only', 'outputs_gt_one_token')}
    counts = Counter(r.get('termination', 'unknown') for r in rows)
    return {'observed_outputs': len(rows), **{key: counts[key] for key in ('eos', 'stop', 'limit', 'fail')},
        'first_prediction_only': sum(r.get('first_prediction_only', r.get('output_tokens', 0) <= 1)
            for r in rows if r.get('termination') in ('eos', 'stop', 'limit')),
        'outputs_gt_one_token': sum(r.get('output_tokens', 0) > 1 for r in rows)}


def task_rows(suite_path, suite, job, folder, config, summary, reader):
    common = base_fields(suite_path, suite, job, config)
    common['summary_status'] = (summary or {}).get('status', 'MISSING')
    entries = {row['arm']: row for row in (summary or {}).get('evaluations', []) if row['task'] == job['unit']}
    raw_outputs = reader.rows(folder/'raw_outputs.jsonl')
    loaded = {}
    for arm, entry in entries.items():
        result_path = folder/Path(entry['file']).name
        result = reader.json(result_path, {})
        from_raw, docs = sample_metrics(result)
        metrics = entry.get('sample_weighted_metrics') or from_raw
        loaded[arm] = (entry, result, result_path, metrics, docs)
    native = loaded.get('native')
    gsm_documents, gsm_pairs, gsm_summaries, gsm_groups = (gsm8k_pairs(common, job, loaded)
        if job['unit'] == 'gsm8k' else ([], [], [], {}))
    gsm_lookup = {(row['arm'], row['filter']): row for row in gsm_summaries}
    metrics_out, leaves = [], []
    for arm in job['arms']:
        if arm not in loaded:
            metrics_out.append({**common, 'arm': arm, 'arm_status': 'NOT_ACCEPTED_OR_NOT_RUN',
                'metric_filter': None, 'value': None, 'native_value': None, 'delta_pp': None,
                'control_delta_pp': None, 'sample_rows': None, 'documents': None,
                'leaf_count': None, 'documents_per_leaf': None, 'result_file': None,
                **termination(None)})
            continue
        entry, result, result_path, metrics, docs = loaded[arm]
        populations = {task: len(ids) for task, ids in docs.items()}
        evidence = 'accepted runner entry; native/candidate sample signature checked by runner'
        for metric, value in metrics.items():
            baseline = native[3].get(metric) if native else None
            paired = baseline is not None and baseline.get('sample_rows') == value.get('sample_rows')
            if native and docs and native[4]:
                paired = paired and docs == native[4]
            current, old = numeric(value.get('mean')), numeric(baseline.get('mean')) if baseline else None
            delta = 100*(current-old) if paired and current is not None and old is not None else None
            if job['unit'] == 'gsm8k':
                metric_name, _, extraction = metric.partition(',')
                comparison = gsm_lookup.get((arm, extraction))
                group = gsm_groups.get((arm, extraction))
                if arm == 'native':
                    paired = metric_name == 'exact_match' and group is not None and not group['errors']
                    delta = 0. if paired else None
                else:
                    paired = metric_name == 'exact_match' and comparison is not None and comparison['pairing_status'] == 'PAIRED_COMPLETE_POPULATION'
                    delta = comparison['paired_accuracy_difference_pp'] if paired else None
                evidence = 'raw official doc_id plus document/prompt/target hashes; complete unique population per filter' if paired else 'raw pairing unavailable or rejected; see gsm8k_paired.csv'
            rows = None if raw_outputs is None else [r for r in raw_outputs if r.get('arm') == f"{job['unit']}.{arm}" and r.get('generate') is True]
            # MC's one-token logprob requests are not natural generations.
            if job['unit'] in MC:
                rows = None
            metrics_out.append({**common, 'arm': arm, 'arm_status': 'ACCEPTED_RESULT',
                'metric_filter': metric, 'value': current, 'native_value': old,
                'delta_pp': delta if job['unit'] not in MC else None,
                'control_delta_pp': delta if job['unit'] in MC else None,
                'sample_rows': value.get('sample_rows'),
                'documents': sum(populations.values()) if populations else None,
                'leaf_count': len(populations) if populations else None,
                'documents_per_leaf': compact(populations) if populations else None,
                'population_pairing': evidence if paired or job['unit'] == 'gsm8k' else 'unavailable or different population',
                'result_file': str(result_path), 'raw_samples_available': bool(docs),
                'harness_seeds': compact({k:v for k,v in result.get('config', {}).items() if 'seed' in k}),
                **termination(rows)})
        reported = result.get('results', entry.get('results', {}))
        for task, values in reported.items():
            for key, value in values.items():
                if numeric(value) is None or key == 'alias' or '_stderr' in key:
                    continue
                metric, separator, extraction_filter = key.partition(',')
                stderr_key = metric+'_stderr'+(separator+extraction_filter if separator else '')
                leaves.append({**common, 'arm': arm, 'task_or_group': task, 'metric_filter': key,
                    'reported_value': value, 'reported_sample_stderr': values.get(stderr_key),
                    'stderr_scope': 'harness sample statistic; not an independent-run repetition or CI',
                    'documents': populations.get(task), 'result_file': str(result_path)})
    return metrics_out, leaves, gsm_documents, gsm_pairs, gsm_summaries


def prompt_records(folder, config, suite, reader):
    saved = reader.json(folder/'prompts.json')
    if saved is not None:
        return saved, str(folder/'prompts.json')
    args = {**suite.get('args', {}), **config.get('args', {})}
    recorded = str(args.get('prompts', ''))
    candidates = [Path(recorded)] if recorded else []
    if '/datasets/' in recorded:
        candidates.append(ROOT/'datasets'/recorded.split('/datasets/', 1)[1])
    elif recorded and not Path(recorded).is_absolute():
        candidates.append(ROOT/recorded)
    expected = config.get('prompt_file_sha256') or suite.get('inputs', {}).get('prompts', {}).get('sha256')
    for path in candidates:
        if path.is_file() and expected and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            records = reader.rows(path)
            limit = config.get('args', {}).get('limit', suite.get('args', {}).get('free_limit'))
            return records[:limit] if limit is not None else records, str(path)
    return [], None


def free_rows(suite_path, suite, job, folder, config, summary, reader):
    common = base_fields(suite_path, suite, job, config)
    common['summary_status'] = (summary or {}).get('status', 'MISSING')
    prompts, population_source = prompt_records(folder, config, suite, reader)
    prompt_map = {str(r['id']): (index, r) for index, r in enumerate(prompts)}
    article_ids = {r.get('source_article_id') for r in prompts}
    planned_articles = len(article_ids) if prompts and None not in article_ids else None
    raw = reader.rows(folder/'outputs.jsonl')
    by_arm = defaultdict(dict)
    for index, row in enumerate(raw or [], 1):
        key = str(row['id']); arm = row['arm']
        if key in by_arm[arm]:
            raise ValueError(f'duplicate free prompt/arm, not a run repetition: {folder} {arm} {key}')
        by_arm[arm][key] = {**row, '_line': index}
    reported = (summary or {}).get('arms', {})
    planned = (summary or {}).get('planned_per_arm', config.get('args', {}).get('limit', suite.get('args', {}).get('free_limit')))
    native = by_arm.get('native', {})
    aggregates, outputs = [], []
    for arm in job['arms']:
        arm_report = reported.get(arm); records = list(by_arm.get(arm, {}).values())
        has_raw = bool(records) or (raw is not None and arm_report is not None and arm_report.get('attempted') == 0)
        success = [r for r in records if r.get('termination') in ('eos', 'stop', 'limit')]
        eligible = [r for r in success if numeric(r.get('repetition', {}).get('repeated_4gram_ratio')) is not None]
        flag_values = [bool(r['repetition']['review_flag']) for r in success if 'review_flag' in r.get('repetition', {})]
        pairs = []; paired_rep = []; paired_flags = []
        observed_articles = set(); unknown_article = False
        for row in records:
            key = str(row['id']); item = prompt_map.get(key)
            source = item[1] if item else {}; article = source.get('source_article_id')
            observed_articles.add(article); unknown_article |= article is None
            old = native.get(key)
            matching = old is not None and row.get('prompt_sha256') is not None and row.get('prompt_sha256') == old.get('prompt_sha256')
            if matching and row.get('prompt_token_sha256') is not None:
                matching = row['prompt_token_sha256'] == old.get('prompt_token_sha256')
            completed_pair = matching and row.get('termination') in ('eos', 'stop', 'limit') and old.get('termination') in ('eos', 'stop', 'limit')
            rep = row.get('repetition', {}); old_rep = old.get('repetition', {}) if old else {}
            if completed_pair:
                pairs.append((row, old))
                if numeric(rep.get('repeated_4gram_ratio')) is not None and numeric(old_rep.get('repeated_4gram_ratio')) is not None:
                    paired_rep.append(rep['repeated_4gram_ratio']-old_rep['repeated_4gram_ratio'])
                if 'review_flag' in rep and 'review_flag' in old_rep:
                    paired_flags.append(int(rep['review_flag'])-int(old_rep['review_flag']))
            outputs.append({**common, 'arm': arm, 'prompt_id': key, 'source_article_id': article,
                'window_index': source.get('window_index'), 'derived_sampling_seed': common['seed']+item[0] if item and common['seed'] is not None else None,
                'termination': row.get('termination'), 'output_tokens': row.get('output_tokens'),
                'repeated_4gram_ratio': rep.get('repeated_4gram_ratio'), 'repeated_8gram_ratio': rep.get('repeated_8gram_ratio'),
                'review_flag': rep.get('review_flag'), 'longest_identical_token_run': rep.get('longest_identical_token_run'),
                'paired_completed_native': bool(completed_pair),
                'same_tokens_as_native': row.get('same_tokens_as_native'),
                'output_token_delta_from_native': row.get('output_token_delta_from_native'),
                'source_file': str(folder/'outputs.jsonl'), 'source_line': row['_line']})
        counts = termination(records if has_raw else None)
        if not has_raw and arm_report is not None:
            observed = arm_report.get('attempted')
            counts.update(observed_outputs=observed, first_prediction_only=arm_report.get('first_prediction_only_count'))
            if observed is not None:
                counts.update({key: arm_report.get('termination_counts', {}).get(key, 0) for key in ('eos', 'stop', 'limit', 'fail')})
        attempted = counts['observed_outputs']
        completed = sum(counts[k] for k in ('eos', 'stop', 'limit')) if all(counts[k] is not None for k in ('eos', 'stop', 'limit')) else None
        status = 'NOT_RUN' if attempted == 0 else 'OBSERVED_PARTIAL' if attempted is not None and attempted != planned else 'OBSERVED_COMPLETE' if attempted is not None else 'MISSING'
        if status == 'OBSERVED_COMPLETE' and counts['fail']:
            status = 'OBSERVED_COMPLETE_WITH_FAILURES'
        reported_attempted = (arm_report or {}).get('attempted')
        progress_difference = attempted-reported_attempted if has_raw and reported_attempted is not None else None
        if progress_difference:
            reader.notes.append(f'{job["id"]}/{arm}: JSONL has {attempted} records; separately read summary reports {reported_attempted}. Raw rows used; snapshot is not atomic.')
        aggregates.append({**common, 'arm': arm, 'arm_status': status, 'planned_windows': planned,
            'planned_source_articles': planned_articles, 'population_source': population_source,
            'observed_source_articles': len(observed_articles) if records and not unknown_article else None,
            'completed_outputs': completed, **counts,
            'unrecorded_planned_windows': planned-attempted if planned is not None and attempted is not None else None,
            'reported_not_run': arm_report.get('not_run') if arm_report else None,
            'reported_attempted': reported_attempted,
            'raw_minus_reported_attempted': progress_difference,
            'completed_output_tokens_mean': mean([r['output_tokens'] for r in success]) if has_raw else (arm_report or {}).get('completed_output_tokens_mean'),
            'mean_repeated_4gram_ratio': mean([r['repetition']['repeated_4gram_ratio'] for r in eligible]) if has_raw else (arm_report or {}).get('mean_repeated_4gram_ratio'),
            'repetition_eligible_completed': len(eligible) if has_raw else None,
            'review_flag_count': sum(flag_values) if flag_values else 0 if has_raw and completed == 0 else (arm_report or {}).get('review_flag_count') if not has_raw else None,
            'review_flag_denominator_completed': len(flag_values) if has_raw else completed,
            'paired_completed_with_native': len(pairs) if has_raw else None,
            'paired_mean_output_token_delta': mean([r['output_tokens']-old['output_tokens'] for r,old in pairs]),
            'paired_repetition_pairs': len(paired_rep) if has_raw else None,
            'paired_mean_repeated4_delta': mean(paired_rep),
            'paired_review_flag_delta_pp': 100*mean(paired_flags) if paired_flags else None,
            'reported_summary': compact(arm_report) if arm_report is not None else None,
            'grouping': 'windows from source articles; neither windows nor tokens are independent system runs',
            'repetition_scope': 'descriptive degeneration screen, not correctness or task accuracy'})
    return aggregates, outputs


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def tex(value):
    if value is None:
        return 'NA'
    if isinstance(value, float):
        return f'{value:.3f}'
    translations = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}'}
    return ''.join(translations.get(c, c) for c in str(value))


def latex(path, headers, rows, note):
    lines = ['% '+note, r'\begin{tabular}{'+'l'*len(headers)+'}', r'\hline', ' & '.join(map(tex, headers))+r' \\', r'\hline']
    lines += [' & '.join(map(tex, row))+r' \\' for row in rows]
    lines += [r'\hline', r'\end{tabular}', '']
    path.write_text('\n'.join(lines))


def tables(out, tasks, free):
    generated = []
    decode = [r for r in tasks if r['scope'] != 'MC_PREFILL_IDENTITY_CONTROL']
    if decode:
        name = 'task_decode_table.tex'
        latex(out/name, ['Job / format', 'Seed', 'Task / arm', 'Metric, filter', 'N', 'Score', 'Delta pp', 'State'],
            [[r['job']+'/'+str(r['format']), r['seed'], r['unit']+'/'+r['arm'], r['metric_filter'], r['sample_rows'], r['value'], r['delta_pp'], r['arm_status']] for r in decode],
            'GSM8K decode generation; first prediction is prefill. Scores retain raw [0,1] units; delta is percentage points. Missing=NA. No run CI.')
        generated.append(name)
    groups = defaultdict(list)
    for row in tasks:
        if row['scope'] == 'MC_PREFILL_IDENTITY_CONTROL':
            groups[(row['suite'], row['job'], row['metric_filter'])].append(row)
    if groups:
        records = []
        for (_, job, metric), rows in groups.items():
            if metric is None:
                continue
            native = next((r for r in rows if r['arm'] == 'native'), {})
            differences = [abs(r['control_delta_pp']) for r in rows if r['control_delta_pp'] is not None]
            records.append([job, rows[0]['seed'], metric, native.get('documents'), native.get('leaf_count'),
                            native.get('value'), f"{len(rows)}/{rows[0]['planned_configurations']}", max(differences) if differences else None])
        if records:
            name = 'mc_identity_controls.tex'
            latex(out/name, ['Job', 'Seed', 'Metric, filter', 'Docs', 'Leaves', 'Native score', 'Accepted configs', 'Max abs drift pp'], records,
                'MC PREFILL-ONLY identity controls, not decode-policy task improvements. No candidate-method accuracy claim. MMLU limit applies per subject. NA is unmeasured.')
            generated.append(name)
    if free:
        name = 'free_generation_table.tex'
        latex(out/name, ['Job / arm', 'Seed', 'Windows / articles', 'Observed', 'EOS / stop / cap / fail', 'Unrecorded', 'Mean 4gram', 'Paired delta 4gram'],
            [[r['job']+'/'+r['arm'], r['seed'], f"{tex(r['planned_windows'])}/{tex(r['planned_source_articles'])}",
              r['observed_outputs'], '/'.join(tex(r[k]) for k in ('eos','stop','limit','fail')),
              r['unrecorded_planned_windows'], r['mean_repeated_4gram_ratio'], r['paired_mean_repeated4_delta']] for r in free],
            'Natural free continuations; windows sharing source_article_id are grouped, not independent articles. Repetition is descriptive, not accuracy. Paired deltas use completed same-prompt native pairs; missing=NA.')
        generated.append(name)
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suite', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('choose an empty output directory to preserve earlier summaries')
    reader = Reader(); tasks = []; leaves = []; free = []; outputs = []; jobs = []
    gsm_documents = []; gsm_pairs = []; gsm_summaries = []
    reader.text(__file__)
    suite_paths = [p/'suite.json' if p.is_dir() else p for p in args.suite]
    if len({p.resolve() for p in suite_paths}) != len(suite_paths):
        parser.error('duplicate suite input')
    for path in suite_paths:
        path = path.resolve(); suite = reader.json(path)
        if not isinstance(suite, dict) or not isinstance(suite.get('jobs'), list):
            raise ValueError(f'expected run_task_quality suite.json: {path}')
        for job in suite['jobs']:
            folder = job_directory(path, job); config = reader.json(folder/'config.json', {})
            summary = reader.json(folder/'summary.json')
            jobs.append({'suite': str(path), 'suite_status': suite.get('status'), 'job': job['id'],
                'job_status': job.get('status'), 'unit': job['unit'], 'format': job.get('format'),
                'expected_arms': job['arms'], 'summary_status': (summary or {}).get('status', 'MISSING')})
            if job['unit'] == 'free_generation':
                aggregate, raw = free_rows(path, suite, job, folder, config, summary, reader)
                free.extend(aggregate); outputs.extend(raw)
            else:
                aggregate, raw, documents, pairs, paired_summary = task_rows(path, suite, job, folder, config, summary, reader)
                tasks.extend(aggregate); leaves.extend(raw)
                gsm_documents.extend(documents); gsm_pairs.extend(pairs); gsm_summaries.extend(paired_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for name, rows in [('task_metrics.csv',tasks), ('task_reported_leaf_metrics.csv',leaves),
                       ('free_generation.csv',free), ('free_outputs.csv',outputs),
                       ('gsm8k_documents.csv',gsm_documents), ('gsm8k_paired_documents.csv',gsm_pairs),
                       ('gsm8k_paired.csv',gsm_summaries)]:
        if rows:
            write_csv(args.output_dir/name, rows); generated.append(name)
    generated.extend(tables(args.output_dir, tasks, free))
    accepted = sum(r.get('value') is not None for r in tasks)+sum(r.get('observed_outputs') is not None and r['observed_outputs']>0 for r in free)
    manifest = {'schema_version':1, 'status':'HAS_MEASURED_RESULTS' if accepted else 'NO_MEASURED_RESULTS',
        'jobs':jobs, 'counts':{'task_metric_rows':len(tasks),'reported_leaf_metric_rows':len(leaves),'free_arm_rows':len(free),'free_output_rows':len(outputs),
            'gsm8k_document_filter_rows':len(gsm_documents), 'gsm8k_paired_document_filter_rows':len(gsm_pairs),
            'gsm8k_paired_arm_filter_rows':len(gsm_summaries),
            'gsm8k_rejected_arm_filter_pairs':sum(row['pairing_status'] != 'PAIRED_COMPLETE_POPULATION' for row in gsm_summaries)},
        'sources':list(reader.sources.values()), 'notes':reader.notes,
        'semantics':['No averaging across suites, seeds, formats, runs or extraction filters.',
            'MC is a prefill-only identity control. Its differences are control drift, not decode-policy improvements.',
            'Accepted standard entries are committed after runner sample-identity checks; unaccepted result files are not promoted to scores.',
            'Missing values are CSV blanks / LaTeX NA, never inferred zero. Recorded zero attempts or zero observed failures remain zero.',
            'Harness stderr describes task samples, not independent system runs; no run confidence intervals are manufactured.',
            'GSM8K correctness/discordance uses raw exact_match, separate strict-match and flexible-extract filters; never pool their duplicated documents.',
            'GSM8K pairing requires unique official doc_ids, identical document/prompt/target hashes and complete harness-effective populations in both arms; missing/mismatched groups have no paired accuracy delta.',
            'Free windows preserve source-article grouping; repetition/length are descriptive and do not measure correctness.',
            'Natural EOS/stop, length cap and observed failures are separate; killed/unrecorded requests are not fabricated as failure outputs.'],
        'generated_files':[*generated,'summary.json']}
    (args.output_dir/'summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:manifest[key] for key in ('status','counts','generated_files')},indent=2))


if __name__ == '__main__':
    main()
