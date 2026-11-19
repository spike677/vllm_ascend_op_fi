"""Freeze one ordinary image acceptance case; no device/model calls."""
import argparse,hashlib,json,shlex
from pathlib import Path
from experiments.c2c3_closure_20261117.prepare_vision17_worker_census import cases as census_cases
from experiments.c2c3_closure_20261117.queue_contract import validate_cases

def cases(source,installed,manifest_sha,card=5):
    source,installed=Path(source),Path(installed)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=source/'source_manifest.json'
    if sha(manifest)!=manifest_sha:raise ValueError('FROZEN_SOURCE_MANIFEST_CHANGED')
    data=json.loads(manifest.read_bytes());matched=0
    for row in data['files']:
        relative=row['path'];path=source/relative
        if not path.resolve().is_relative_to(source.resolve()) or sha(path)!=row['sha256']:
            raise ValueError('FROZEN_SOURCE_CHANGED:'+relative)
        target=(relative.removeprefix('src/') if relative.startswith('src/demoncat/')
            else relative if relative.startswith('fault_injection/') and relative.endswith('.py') else None)
        if target:
            if sha(installed/target)!=row['sha256']:raise ValueError('INSTALLED_SOURCE_CHANGED:'+target)
            matched+=1
    row=census_cases(str(source),card)[0]
    row['id']='vision17-ordinary-two-tasks'
    row['mounts'].append([str(installed),'/installed','ro'])
    row['env']['PYTHONPATH']='/installed:/source:/source/src:/source/fault39/src'
    row['command']='source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '+shlex.join([
        'python3','-B','-u','/source/experiments/worker_core_v11_20261118/vision17_task_probe.py',
        '--model','/model','--image','/natural-input.JPG','--out','/evidence/probe','--memory','.60',
        '--source-manifest-sha256',manifest_sha])
    row['timeout']=1200
    row['review']['resource_budget'].update(timeout_seconds=1200,nominal_actual_requests=8,max_actual_requests_with_recovery=10)
    row['review'].update(experiment_id=row['id'],provider='Vision17TaskProvider via public Host/Worker',
        source_commit=data['base_commit'],source_manifest_sha256=manifest_sha,installed_python_sources_matched=matched,
        expected_hits=2,acceptance='TWO_INDEPENDENT_ORDINARY_FOUR_REQUEST_TASKS_TORCH_PARAMETER_ONLY',
        recovery_plan='Each normal/F/OFF/health uses a fresh actual single-image encoder request. Public idle cache reset only. '
            'Failure preserves first error and ownership until actual OFF/health; no F retry.',
        statistics=False,dump=False,full_census=False,C1=False,C2=False,C3=False,E5=False,
        raw_evidence=['/evidence/probe/result.json','/evidence/process.log'])
    return validate_cases([row],[card])

def main():
    p=argparse.ArgumentParser()
    for key in ('source','installed','source-manifest-sha256','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--card',type=int,choices=(2,5),default=5);a=p.parse_args()
    rows=cases(a.source,a.installed,a.source_manifest_sha256,a.card)
    with Path(a.out).open('x') as f:json.dump(rows,f,indent=2)
if __name__=='__main__':main()
