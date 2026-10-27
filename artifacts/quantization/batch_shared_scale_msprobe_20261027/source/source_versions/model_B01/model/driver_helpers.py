"""One-load native INT8 TP2 screen + objective batch/controlled fault matrix.

Fault requires --allow-fault AND inherited BATCH_OUTLIER_ENABLED=1. No remote
execution. Each generation freezes real tokenizer IDs, then binds first actual
all-request pure-decode call at the selected layer on each TP rank.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from prompt_cases import CASES,score


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def events(root,generation):
    result=[]
    for p in (root/'operators').glob('pid*.jsonl'):
        for line in p.read_text(encoding='utf-8').splitlines():
            e=json.loads(line)
            if e.get('generation_id')==generation:result.append(e)
    return result


def certify(root,generation,expected_logical):
    ev=events(root,generation)
    calls=[x for x in ev if x['event']=='native_bound_call']
    if len(calls)!=2 or {x['rank'] for x in calls}!={0,1}:
        raise RuntimeError('Expected exactly one actual bound native call per rank: '+generation)
    expected={}
    row_order=None
    for c in calls:
        b=c['binding_event'];s=c['scheduler']
        assert b['status']=='COMPLETE' and b['input_unchanged']
        assert 1<=len(b['consumers'])<=4 and all(x['same_q_object'] and x['same_scale_object'] for x in b['consumers'])
        assert b['non_target_pair_preserved'] and s['pure_decode'] and s['all_expected_present']
        assert set(s['logical_ids'])==set(expected_logical) and len(s['logical_ids'])==len(expected_logical)
        assert s['logits_indices']==list(range(len(expected_logical)))
        if row_order is None:row_order=s['logical_ids']
        assert row_order==s['logical_ids'],'TP ranks disagree on logical request rows'
        expected[str(c['rank'])]=c['contract']
    l=[x for x in ev if x['event']=='first_logits']
    if len(l)!=2 or {x['rank'] for x in l}!={0,1}:raise RuntimeError('Missing actual logits on one TP rank')
    return {'contracts':expected,'calls':calls,'logits':{str(x['rank']):x['tensor'] for x in l},
            'stats':[x for x in ev if x['event']=='activation_statistics']}


def signatures(requests):
    return {x['logical_id']:{k:x[k] for k in ('token_ids','text','logprobs','finish_reason')} for x in requests}


def healthy_equal(a,b):
    def pairs(run):
        out={}
        for c in run['proof']['calls']:
            e=c['binding_event']
            out[str(c['rank'])]={k:e[k] for k in ('native_q_sha256','native_scale_sha256','q_sha256','scale_sha256')}
            out[str(c['rank'])]['consumer_output_sha256']=[v['output_sha256'] for v in e['consumers']]
        return out
    return a['proof']['contracts']==b['proof']['contracts'] and signatures(a['requests'])==signatures(b['requests']) and all(
        a['proof']['logits'][r]['logical_sha256']==b['proof']['logits'][r]['logical_sha256'] for r in ('0','1')) and pairs(a)==pairs(b)


