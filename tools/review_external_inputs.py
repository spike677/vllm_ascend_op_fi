"""Cross-check real product requests against the input examples' raw call records."""
import argparse
import json
from pathlib import Path
from tools.review_product_mode import review


def check(data, records):
    result=review(data)
    events={}
    for record in records:
        for event in record['events']:
            if len(event.get('request_ids',[]))!=1:raise ValueError('INPUT_REQUEST_OWNER_AMBIGUOUS')
            key=(record['rank'],event['request_ids'][0])
            events.setdefault(key,[]).append(event)
    consumed=set()
    for scope in data['scopes']:
        for request in scope['requests'][:3]:
            for rank in request['result']['ranks']:
                key=(rank['rank'],rank['request_id']);calls=events.get(key,[]);consumed.add(key)
                if not calls or len(calls)!=rank['coverage']['eligible']:raise ValueError('INPUT_CALL_COUNT_MISMATCH')
                for event in calls:
                    if not all(event.get(k) for k in ('applied','consumed','restored','consumer_operator')):
                        raise ValueError('INPUT_CONSUMPTION_OR_RESTORE_MISSING')
                    if event.get('failed') or event['before_sha256']!=event['restored_sha256']:
                        raise ValueError('INPUT_BYTES_NOT_RESTORED')
                    if not 0<=event['changed_elements']<=2 or event['changed']!=int(event['changed_elements']>0):
                        raise ValueError('INPUT_CHANGE_COUNT_INVALID')
                    if data['mechanism']=='external.module_input_gain':
                        if event.get('whole_input_restored') is not True:raise ValueError('WHOLE_INPUT_NOT_RESTORED')
                    elif not (event.get('argument_substitution') and event.get('original_input_unchanged')
                              and event.get('same_stream_completed')):raise ValueError('ARGUMENT_PROOF_MISSING')
    if set(events)!=consumed:raise ValueError('UNASSOCIATED_INPUT_CALLS')
    return dict(product=result,call_records=sum(map(len,events.values())),status='INPUT_SITES_REVIEW_PASSED',qualification_granted=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('receipt');p.add_argument('--output');a=p.parse_args()
    path=Path(a.receipt);records=[]
    for f in sorted(path.parent.glob('external-input-rank*.jsonl')):
        records.extend(json.loads(line) for line in f.read_text().splitlines())
    result=check(json.loads(path.read_bytes()),records);text=json.dumps(result,indent=2)+'\n'
    if a.output:Path(a.output).write_text(text)
    else:print(text)


if __name__=='__main__':main()
