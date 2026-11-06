"""New CPU continuation helpers; never overwrite the frozen original receipts."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import uuid

from generation import layout

ROOT=Path('/home/fault33-paged-tiling-20261105')
IMAGE='sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9'
NEW_SOURCES=('resume_support.py','resume_cpu.py','packaging_preflight.py','resume_stage_cpu.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')


def attempt_paths(generation,attempt):
    paths=layout(ROOT,generation)
    if not re.fullmatch(r'a[1-9][0-9]{0,2}',attempt):
        raise ValueError('Attempt must be a1..a999')
    return paths,paths['evidence']/f'resume_{attempt}'


def verify_original(paths):
    base=paths['evidence']
    freeze=json.loads((base/'source_freeze.json').read_text())
    changed=[p for p,digest in freeze['files'].items() if not (ROOT/p).is_file() or sha(ROOT/p)!=digest]
    if changed:
        raise RuntimeError('Original frozen source changed: '+str(changed))
    failure=json.loads((base/'build_completion.json').read_text())
    if failure['status']!='FAIL_CPU_BUILD' or failure['returncode']!=2 or not failure['container_removed'] or failure['frozen_sources_changed']:
        raise RuntimeError('Original failed CPU build is not the reviewed clean continuation base')
    if failure['log_sha256']!=sha(base/'build.log'):
        raise RuntimeError('Original failure log changed')
    return freeze


def new_source_freeze():
    return {name:sha(ROOT/'source_policy'/name) for name in NEW_SOURCES}


def verify_new_source_freeze(expected):
    if new_source_freeze()!=expected:
        raise RuntimeError('Continuation sources changed')


def run_cpu(script,evidence,name,timeout):
    """Bounded own-ID cleanup only; no devices, privileges or outside writes."""
    token=uuid.uuid4().hex
    cidfile=evidence/(name+'.cid')
    argv=['docker','run','--rm','--network','none','--cpus','4','--memory','16g','--pids-limit','512',
        '--name',name,'--label','fault33.cpu_resume_token='+token,'--cidfile',str(cidfile),
        '-v',str(ROOT)+':/audit','-v',str(ROOT/f'dependencies_{evidence.parent.name.removeprefix("evidence_")}')+':/deps',
        '-v','/usr/bin/patch:/buildtools/patch:ro','--entrypoint','bash',IMAGE,'-lc',script]
    start={'argv':argv,'started_unix':time.time(),'timeout_seconds':timeout,'devices':[],'privileged':False,'network_enabled':False}
    save(evidence/(name+'_start.json'),start)
    logpath=evidence/(name+'.log')
    rc=None
    with logpath.open('x') as log:
        try:
            rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc=124
            if cidfile.is_file():
                cid=cidfile.read_text().strip()
                if not re.fullmatch('[0-9a-f]{64}',cid):
                    raise RuntimeError('Invalid owned CPU container ID')
                inspected=subprocess.run(['docker','inspect',cid],capture_output=True,text=True)
                if inspected.returncode==0:
                    obj=json.loads(inspected.stdout)[0]
                    if obj['Image']!=IMAGE or obj['Name'].lstrip('/')!=name or obj['Config']['Labels'].get('fault33.cpu_resume_token')!=token:
                        raise RuntimeError('Refuse cleanup of an unproven container')
                    subprocess.run(['docker','stop','--time','5',cid],check=True)
    cid=cidfile.read_text().strip() if cidfile.is_file() else None
    removed=cid is not None and subprocess.run(['docker','inspect',cid],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
    return {'returncode':rc,'container_removed':removed,'finished_unix':time.time(),'log_sha256':sha(logpath),
        'log_path':str(logpath),'container_id':cid,'npu_executed':False,'global_sdk_modified':False}
