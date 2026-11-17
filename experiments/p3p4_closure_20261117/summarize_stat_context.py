"""Resolve an already selected statistics first difference using raw context only."""
import argparse
import hashlib
import json
from pathlib import Path


def context(folder,operator):
    stack=json.loads((folder/'stack.json').read_bytes())
    construct=json.loads((folder/'construct.json').read_bytes())
    matches=[value for value in stack.values()
        if isinstance(value,list) and len(value)==2 and isinstance(value[0],list) and operator in value[0]]
    if len(matches)!=1 or operator not in construct:raise RuntimeError('MISSING_OR_AMBIGUOUS_RAW_CONTEXT')
    return construct[operator],matches[0][1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
    for path in a.reports:
        report=json.loads(path.read_bytes());comparison=report['comparison']['comparison']
        row=dict(report=str(path),report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status=comparison['status'])
        if comparison.get('identity'):
            operator=comparison['identity'].split('/')[0]
            b=context(Path(report['baseline']),operator);f=context(Path(report['fault']),operator)
            if b!=f:raise RuntimeError('B_F_SOURCE_CONTEXT_DIFFERS')
            row.update(identity=comparison['identity'],construct_parent=b[0],unique_stack_entry=True,
                top_frame=b[1][0] if b[1] else None,
                model_frames=[line for line in b[1] if '/transformers/models/' in line],
                entire_stack_sha256=hashlib.sha256(json.dumps(b[1],ensure_ascii=False).encode()).hexdigest())
        rows.append(row)
    result=dict(source='RAW_STACK_CONSTRUCT_ONLY',native_logs_read=False,private_tensors_read=False,
        selects_first_difference=False,rows=rows)
    with a.out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(out=str(a.out),rows=len(rows))))


if __name__=='__main__':main()
