"""Read-only numerical neutrality and scalar dump inventory for a finished case."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path('/home/fault28-fia-grain-20261109')
OLD=Path('/home/fault28-fia-graph-20261106')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True)
    p.add_argument('--suffix',default='grain_a1');a=p.parse_args();name=a.case
    new=ROOT/'outputs'/(name+'_'+a.suffix)/name;old=OLD/'outputs'/(name+'_fp64_a1')/name
    n=json.loads((new/'result.json').read_bytes());o=json.loads((old/'result.json').read_bytes())
    assert n['status']==o['status']=='CAPTURE_COMPLETED' and len(n['calls'])==len(o['calls'])==3
    for x,y in zip(n['calls'],o['calls']):
        for key in ('full_sha256','target_sha256','request_sha256','shape'):assert x[key]==y[key],key
        for key in ('full_sha256','target_sha256','request_sha256','shape'):assert x['consumer'][key]==y['consumer'][key],key
        for key in ('target_file',):assert (new/x[key]).read_bytes()==(old/y[key]).read_bytes()
        assert (new/x['consumer']['target_file']).read_bytes()==(old/y['consumer']['target_file']).read_bytes()
    files=[f for f in new.rglob('*') if f.is_file()]
    interesting=[{'file':f.relative_to(new).as_posix(),'bytes':f.stat().st_size} for f in files
                 if f.suffix not in ('.json','.jsonl','.csv','.txt') and 'private_output' not in f.parts]
    log=ROOT/'outputs'/(name+'_'+a.suffix)/(name+'.log')
    traces=[]
    with log.open(errors='replace') as stream:
        for number,line in enumerate(stream,1):
            if 'F28_GRAIN_V1' in line:traces.append({'line':number,'text':line.strip()[:1000]})
    print(json.dumps({'status':'PASS_PILOT_NUMERICAL_NEUTRALITY','case':name,'calls':3,
                     'graph_output_hashes_and_complete_private_targets_unchanged':True,
                     'file_extensions':dict(Counter(f.suffix for f in files)),
                     'scalar_dump_candidates':interesting[:30],'stdout_trace_count':len(traces),
                     'stdout_trace_examples':traces[:4]},indent=2))


if __name__=='__main__':main()
