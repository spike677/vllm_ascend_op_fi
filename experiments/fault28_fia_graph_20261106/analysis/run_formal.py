"""Serial finite cases; each retains controller lock/gates/timeout/cleanup."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/fault28-fia-graph-20261106')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', choices=('fp32','fp64'), default='fp32')
    p.add_argument('--cases',nargs='+',required=True);p.add_argument('--suffix',required=True)
    a=p.parse_args()
    assert 1<=len(a.cases)<=8 and len(set(a.cases))==len(a.cases)
    matrix=json.loads((ROOT/('setup/fp64_matrix_v1.json' if a.collection=='fp64' else 'setup/formal_matrix_v3.json')).read_bytes())
    assert all(name in matrix['case_ids'] for name in a.cases)
    for name in a.cases:
        path=ROOT/('setup/fp64_v1' if a.collection=='fp64' else 'setup')/(name+'.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==matrix['case_sha256'][name]
        command=[sys.executable,'-B',str(ROOT/'analysis/launch_graph_case.py'),'--case',name,
                 '--collection',a.collection,'--attempt',name+'_'+a.suffix,'--capture','msprof','--clear-card5-atk']
        print(json.dumps({'event':'BEGIN','case':name,'command':command}),flush=True)
        rc=subprocess.call(command)
        if rc: return rc
        print(json.dumps({'event':'COMPLETED','case':name}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
