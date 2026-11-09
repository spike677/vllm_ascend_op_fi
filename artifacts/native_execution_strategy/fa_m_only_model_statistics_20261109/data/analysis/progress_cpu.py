"""Read-only partial/completed model status; never imports a framework."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT=Path('/home/fault28-m-only-model-20261109')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--case',required=True);a=p.parse_args()
    root=ROOT/'outputs'/a.attempt;folder=root/a.case
    result={}
    if (root/'completion.json').exists():
        done=json.loads((root/'completion.json').read_bytes())
        result.update(controller_status=done['status'],controller_error=done.get('traceback'),cleanup_errors=done['cleanup_errors'])
    if (folder/'result.json').exists():
        r=json.loads((folder/'result.json').read_bytes())
        result.update(status=r['status'],completed_phases=r.get('completed_phases'),calls=len(r['calls']),error=r.get('traceback'))
    snapshots=sorted((folder/'native_calls').glob('call_*/call.json'))
    result['completed_call_sidecars']=len(snapshots)
    if snapshots:
        last=json.loads(snapshots[-1].read_bytes())
        result['last_completed_call']={key:last[key] for key in ('step','stage','mode','layer_index')}
    if (folder/'phase_manifest.json').exists():
        phases=json.loads((folder/'phase_manifest.json').read_bytes())['phases'];rows=[]
        for phase in phases:
            stats={'step':phase['step'],'stage':phase['stage'],'mode':phase['mode'],
                   'logits_sha256':phase['logits']['full_sha256']}
            observation=phase['logits'].get('exact_numeric_observation',{})
            if 'comparison' in observation:stats['changed_logits']=observation['comparison']['changed_elements']
            rows.append(stats)
        result['phases']=rows
    print(json.dumps(result,indent=2))
