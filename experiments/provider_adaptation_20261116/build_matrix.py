"""Host CPU controller: no NPU devices, no foreign containers, no library loading."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

VERSIONS={'851':'8.5.1','90':'9.0.0','91':'9.1.0','92':'9.2.0'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
    if not a.attempt.isalnum():p.error('alphanumeric attempt required')
    root=Path('/home/provider-adaptation-20261116')
    def build(item):
        tag,version=item;name=f'provider-build-{tag}-20261116'
        info=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
        if info['Config']['Labels'].get('dcat.task')!='provider-adaptation-20261116':raise RuntimeError('FOREIGN_CONTAINER')
        host=info['HostConfig']
        if host['Privileged'] or host.get('Devices') or host.get('DeviceRequests'):raise RuntimeError('CPU_CONTAINER_DEVICE_ACCESS')
        output=f'build-{tag}-{a.attempt}'
        command=['docker','exec','-e',f'FAULT39_ASCEND_TOOLKIT=/usr/local/Ascend/cann-{version}/aarch64-linux',
                 '-e',f'FAULT39_OUTPUT_DIR=/work/{output}','-e','FAULT39_PATCHELF=/work/bin/patchelf',
                 name,'bash',f'/work/source-{a.attempt}/fault39/native/build.sh']
        log=root/(output+'.log')
        with log.open('xb') as stream:r=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        result=dict(version=version,image_id=info['Image'],container=name,command=command,returncode=r.returncode,
                    log=log.name,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),actual_npu_calls=0,
                    status='PASS_CPU_BUILD_ONLY' if r.returncode==0 else 'BUILD_FAILED')
        if not r.returncode:
            profile=json.loads((root/output/'version_profile.json').read_text())
            result['profile_sha256']=profile['profile_sha256']
            # This pure file check proves the chosen image matches the build.
            check=subprocess.run(['docker','exec',name,'bash','-c','source "$1" >&2; shift; exec python3 "$@"',
                                  'fault39-cpu-preflight',profile['environment_script']['path'],
                                  f'/work/source-{a.attempt}/fault39/tools/deployment_profile.py',
                                  '--native',f'/work/{output}','--verify-installed','--verify-environment'],capture_output=True,text=True)
            result['installed_check']=dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr)
            if check.returncode:result['status']='INSTALLATION_MISMATCH'
        with (root/(output+'-receipt.json')).open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps(result),flush=True);return result
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(build,VERSIONS.items()))
    return 0 if all(r['status']=='PASS_CPU_BUILD_ONLY' for r in results) else 1


if __name__=='__main__':raise SystemExit(main())
