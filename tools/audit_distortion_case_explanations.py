"""Derive an auditable case explanation from FULL replayed responses, no NPU."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('result', type=Path)
    p.add_argument('--raw-zip', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = json.loads(a.result.read_text(encoding='utf-8'))
    selected, verified = [], 0
    for cycle in result['distortion']['cycles']:
        for row in cycle['matrix']:
            for phase in ('baseline', 'fault', 'recovery'):
                value = row[phase]['full_text']
                assert hashlib.sha256(value.encode()).hexdigest() == row[phase]['response_sha256']
                verified += 1
            if row['probe_id'] not in ('gsm8k-test-0001', 'gsm8k-test-0002', 'gsm8k-test-0004', 'gsm8k-test-0008'):
                continue
            selected.append({'cycle_id': cycle['cycle_id'], 'case_id': row['probe_id'],
                'task_outcomes': row['task_outcomes'], 'visible_distortion': row['visible_distortion'],
                'output_pathology': row['effect']['output_pathology'], 'mechanism': row['mechanism']['status'],
                'recovery_validator_restored': row['task_validator_recovered'],
                'responses': {phase: {**row[phase], 'hash_marker_count': len(re.findall(r'####\s*\d+', row[phase]['full_text']))}
                              for phase in ('baseline', 'fault', 'recovery')},
                'detectors': row['effect']['detectors']})
    audit = {'scorer': result['scorer'], 'raw_zip_sha256': hashlib.sha256(a.raw_zip.read_bytes()).hexdigest(),
             'full_response_sha256_verified_count': verified, 'selected_case_audit': selected,
             'all_case_replay_not_manual_recount': True,
             'live_model_rerun': False, 'old_completion_protocol_limitation_preserved': True}
    with a.output.open('x', encoding='utf-8') as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2); handle.write('\n')
    print(json.dumps({'verified_full_responses': verified, 'selected': len(selected),
                      'raw_zip_sha256': audit['raw_zip_sha256']}))


if __name__ == '__main__':
    main()
