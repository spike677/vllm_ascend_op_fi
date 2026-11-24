"""Review controlled stale-allocation resilience, not natural fault prevalence."""
import argparse
import hashlib
import json
from pathlib import Path

from tools.review_alias15_product import review as mechanism_review
from tools.review_alias15_finiteness import link_calls


def observations(rows, request_id, rank, kind):
    found = [r for r in rows if r.get('request_ids') == [request_id]
             and r.get('rank') == rank and r.get('kind') == kind]
    return dict(records=len(found), nonfinite_records=sum(r['nonfinite'] > 0 for r in found),
                nonfinite_elements=sum(r['nonfinite'] for r in found))


def review(receipt, rows, seeds):
    base = mechanism_review(receipt)
    result = []
    for scope in receipt['scopes']:
        for request in scope['requests'][:3]:
            rid = request['result']['request_id']
            logits_nonfinite = 0
            for r in request['result']['ranks']:
                rank = r['rank']
                seed = [s for s in seeds if s['rank'] == rank and s['request_ids'] == [rid]]
                if (len(seed) != 1 or seed[0]['reused'] is not True
                        or seed[0]['actual_base_written_by_stress'] is not False
                        or seed[0]['actual_nonfinite'] <= 0):
                    raise ValueError('CONTROLLED_REUSE_NOT_PROVEN')
                proof = {k:observations(rows, rid, rank, k)
                         for k in ('projection_input', 'model_hidden', 'model_logits')}
                linked = link_calls(r['alias_detach_effect']['calls'], rows, rank, rid)
                if any(proof[k]['nonfinite_records'] == 0 for k in ('projection_input', 'model_hidden')):
                    raise ValueError('NONFINITE_CONSUMER_OR_HIDDEN_NOT_OBSERVED')
                logits_nonfinite += proof['model_logits']['nonfinite_records']
                result.append(dict(scope=scope['scope'], rank=rank, request_id=rid,
                                   coverage=r['coverage'], observations=proof,
                                   linked_call_evidence=linked,
                                   http_request_completed=r['completed']))
            if logits_nonfinite == 0:
                raise ValueError('NONFINITE_LOGITS_NOT_OBSERVED_ON_ANY_RANK')
    return dict(schema='dcat.alias15.nonfinite_safety.v1',
                status='CONTROLLED_NONFINITE_TWO_OWNER_REVIEW_PASSED',
                scope='A2/0.26/CANN9.1/Qwen3.8-community-W8A8/TP2',
                mechanism_review=base, requests=result,
                allocator_precondition='SEPARATE_FREED_ALLOCATION_NAN_PRECONDITION',
                natural_nonfinite_origin_proven=False, qualification_granted=False,
                limitation='Controlled allocator state; not proof of all nonfinite states or A3 safety')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    a=p.parse_args()
    root=a.directory
    paths=[root/'product_http_15.json']
    rows=[];seeds=[]
    for rank in (0,1):
        for target,pattern in ((rows,'alias15-finite-rank{}.jsonl'),
                               (seeds,'alias15-allocator-stress-rank{}.jsonl')):
            file=root/pattern.format(rank);paths.append(file)
            target.extend(json.loads(line) for line in file.read_text().splitlines())
    result=review(json.loads(paths[0].read_bytes()),rows,seeds)
    result['sources']=[dict(file=f.name,bytes=f.stat().st_size,
                           sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in paths]
    result['reviewer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
