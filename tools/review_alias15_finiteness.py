"""Link read-only finite-value observations to actual Alias15 requests/calls."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from tools.review_alias15_product import review


def link_calls(calls, observations, rank, request_id):
    groups = defaultdict(list)
    for row in observations:
        if row['rank'] == rank and row['request_ids'] == [request_id]:
            groups[(row['kind'], row['site'])].append(row)
    used = defaultdict(int)
    result = dict(calls=len(calls), initial_nonfinite=0,
                  base_nonfinite_calls=0, projection_nonfinite_calls=0)
    for call in calls:
        result['initial_nonfinite'] += int(not call['initial_all_finite'])
        for kind, field, count in (
            ('attention_base_return', 'after_sha256', 'base_nonfinite_calls'),
            ('projection_input', 'consumer_sha256', 'projection_nonfinite_calls'),
        ):
            key = (kind, call['site']); index = used[key]
            if index >= len(groups[key]):
                raise ValueError('ALIAS_FINITE_OBSERVATION_MISSING')
            row = groups[key][index]; used[key] += 1
            if row['sha256'] != call[field]:
                raise ValueError('ALIAS_FINITE_VALUE_IDENTITY_MISMATCH')
            if not 0 <= row['nonfinite'] <= row['elements']:
                raise ValueError('ALIAS_FINITE_COUNT_INVALID')
            result[count] += int(row['nonfinite'] > 0)
    if any(len(rows) != used[key] for key, rows in groups.items()
           if key[0] in ('attention_base_return', 'projection_input')):
        raise ValueError('ALIAS_FINITE_UNASSOCIATED_CALL')
    hidden = groups[('model_hidden', 'model')]
    if not hidden:
        raise ValueError('ALIAS_FINITE_HIDDEN_MISSING')
    result.update(hidden_observations=len(hidden),
                  hidden_nonfinite_observations=sum(r['nonfinite'] > 0 for r in hidden),
                  hidden_max_nonfinite=max(r['nonfinite'] for r in hidden),
                  hidden_sha256=[r['sha256'] for r in hidden])
    return result


def check(data, observations):
    base = review(data); requests = []
    for scope in data['scopes']:
        for request in scope['requests'][:3]:
            for rank in request['result']['ranks']:
                requests.append(dict(scope=scope['scope'], rank=rank['rank'],
                    request_id=rank['request_id'], **link_calls(
                        rank['alias_detach_effect']['calls'], observations,
                        rank['rank'], rank['request_id'])))
    return dict(status='ALIAS15_READONLY_FINITE_REVIEW_PASSED', base=base,
                requests=requests, fault_logits_finiteness='NOT_OBSERVED',
                allocation_previous_owner='NOT_OBSERVED', qualification_granted=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('receipt');p.add_argument('--output');a=p.parse_args()
    path=Path(a.receipt);rows=[]
    for file in sorted(path.parent.glob('alias15-finite-rank*.jsonl')):
        rows.extend(json.loads(line) for line in file.read_bytes().splitlines())
    result=check(json.loads(path.read_bytes()),rows)
    raw=(json.dumps(result,indent=2)+'\n').encode()
    if a.output:Path(a.output).write_bytes(raw)
    else:print(raw.decode(),end='')


if __name__=='__main__':main()
