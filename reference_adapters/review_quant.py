"""Review public receipts plus adapter sidecars, without granting qualification."""
import argparse
import hashlib
import json
from pathlib import Path
from tools.review_product_mode import review


def require(value, code):
    if not value:
        raise ValueError(code)


def review_directory(directory):
    directory = Path(directory)
    path = directory / 'product_http_18.json'
    data = json.loads(path.read_bytes())
    public = review(data)
    negative = json.loads((directory / 'product_control_negative.json').read_bytes())
    require(negative['missing_permit_cli']['exit_code'] == 4
        and negative['unknown_profile_cli']['exit_code'] == 4, 'REFERENCE_NEGATIVE_NOT_REJECTED')
    require(data['scopes'][0]['duplicate_enable']['exit_code'] == 4,
        'REFERENCE_DUPLICATE_ENABLE_NOT_REJECTED')
    raw = {}
    sidecar_sha = {}
    for rank in range(2):
        file = directory / f'external-input-rank{rank}.jsonl'
        content = file.read_bytes()
        sidecar_sha[file.name] = hashlib.sha256(content).hexdigest()
        for line in content.splitlines():
            chunk = json.loads(line)
            require(chunk['rank'] == rank, 'REFERENCE_SIDECAR_RANK_MISMATCH')
            for event in chunk['events']:
                require(len(event['request_ids']) == 1, 'REFERENCE_REQUEST_AMBIGUOUS')
                raw.setdefault((rank, event['request_ids'][0]), []).append(event)
    totals = {r: dict(calls=0, changed_calls=0, changed_elements=0, qmm_reads=0,
        saturated_elements=0, clipped_elements=0, zero_elements=0, native_zero_elements=0) for r in range(2)}
    matched = set()
    for scope in data['scopes']:
        for request in scope['requests'][:3]:
            for rank in request['result']['ranks']:
                key = rank['rank'], rank['request_id']
                events = raw[key]; matched.add(key)
                require(len(events) == len(rank['sites']) == rank['coverage']['applied'],
                    'REFERENCE_PUBLIC_SIDECAR_COUNT_MISMATCH')
                for event, site in zip(events, rank['sites']):
                    require(all(event[k] == site[k] for k in ('site', 'changed', 'changed_elements',
                        'before_sha256', 'restored_sha256', 'consumer_operator', 'consumed', 'restored')),
                        'REFERENCE_PUBLIC_SIDECAR_FIELD_MISMATCH')
                    require(event['qmm_reads'] > 0 and event['consumer_operator'] == 'npu::npu_quant_matmul'
                        and event['after_sha256'] == event['consumer_sha256']
                        and event['before_sha256'] == event['restored_sha256'], 'REFERENCE_REAL_QMM_PROOF_MISSING')
                    require(event['scale_multiplier'] in (.25, .5, 2., 4.)
                        and event['rounding'] == 'nearest_even' and event['signed_range'] == [-127, 127],
                        'REFERENCE_SCALE_RECIPE_MISMATCH')
                    sums = totals[rank['rank']]
                    sums['calls'] += 1
                    sums['changed_calls'] += event['changed']
                    for k in ('changed_elements', 'qmm_reads', 'saturated_elements', 'clipped_elements',
                              'zero_elements', 'native_zero_elements'):
                        require(type(event[k]) is int and event[k] >= 0, 'REFERENCE_INVALID_EFFECT_COUNT')
                        sums[k] += event[k]
    require(matched == set(raw), 'REFERENCE_ORPHAN_SIDECAR_REQUEST')
    for sums in totals.values():
        require(0 < sums['changed_calls'] <= sums['calls'], 'REFERENCE_NO_EFFECT')
        if 'row9' in data['mechanism']:
            require(sums['clipped_elements'] > 0 and sums['saturated_elements'] > 0,
                'REFERENCE_SATURATION_NOT_OBSERVED')
        else:
            require(sums['zero_elements'] > sums['native_zero_elements'], 'REFERENCE_SQUEEZE_NOT_OBSERVED')
    return dict(status='REFERENCE_A2_TP2_TWO_OWNERS_REVIEWED', mechanism=data['mechanism'],
        semantics='PHENOMENON_PROXY_NOT_ORIGINAL_MECHANISM', public_review=public,
        effects_by_rank=totals, sidecar_sha256=sidecar_sha,
        receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), qualification_granted=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = review_directory(args.directory)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'mechanism', 'effects_by_rank')}))


if __name__ == '__main__':
    main()
