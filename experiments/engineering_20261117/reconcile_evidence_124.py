"""Bind real saved phases and raw statistics to one append-only audit scope."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sys.path[:0]=[str(a.source),str(a.source/'fault39/src')]
    from fault_injection.evidence import EvidenceManager,canonical_sha,digest
    result=json.loads((a.case/'result.json').read_bytes());state=result['provider'];q=state['qualification']
    candidates=q['candidate_set']['candidates']
    if len(candidates)!=1:raise ValueError('UNIQUE_REAL_SCOPE_REQUIRED')
    target=candidates[0]['identity']
    env={key:result[key] for key in ('torch','torch_npu','device','scope','pretrained_model','public_vllm_worker')}
    scope=dict(mechanism=result['mechanism'],provider_version='1',
        model_instance_id=target['model_instance_id'],worker_start_identity=target['worker_id'],
        process_start_identity=target['process_start_identity'],execution_boundary='python_hook',
        operator=target['operator_api'],owner=state['owner'],generation=state['generation'],
        pid=target['process_id'],rank=target['rank'],logical_device=target['logical_device'],
        source_sha256=digest(a.source/'source_manifest.json'),environment_sha256=canonical_sha(env),
        scope_kind=target['scope_kind'],reconciled_saved_run=True,runtime_admission_granted=False)
    evidence=EvidenceManager(a.out,scope)
    evidence.write('environment',env);evidence.write('candidate_set',q['candidate_set'])
    for phase in q['phases']:evidence.write(phase['phase'],phase)
    evidence.write('recovery',result['cleanup']);evidence.reference('runtime_result',a.case/'result.json')
    capture=a.case/'statistics'
    for name,step in (('baseline',0),('fault',3)):
        paths=list((capture/'observation'/f'step{step}').rglob('dump.json'))
        if len(paths)!=1:raise ValueError('AMBIGUOUS_RAW_STATISTICS')
        evidence.statistics(name,step,0,paths[0].parent,config=capture/'config.json',
                            collector=capture/'collector_identity.json')
    evidence.reference('independent_analysis',a.analysis)
    manifest=evidence.finalize()
    print(json.dumps(dict(status='SAVED_RUN_RECONCILED_NOT_NEW_ACCEPTANCE',
                         scope_sha256=evidence.scope_sha256,manifest_sha256=digest(manifest),out=str(a.out))))


if __name__=='__main__':main()
