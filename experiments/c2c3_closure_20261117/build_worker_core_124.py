"""Root-owned, device-free wheels and observer build from one frozen source."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid


IMAGE = 'sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve(strict=True)
    manifest=json.loads((source/'source_manifest.json').read_bytes())
    for row in manifest['files']:
        path=source/row['path']
        if path.is_symlink() or not path.resolve().is_relative_to(source) or sha(path)!=row['sha256']:
            raise RuntimeError('FROZEN_SOURCE_CHANGED:'+row['path'])
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'cann851').mkdir()
    owner=uuid.uuid4().hex
    script='''set -e
cp -a /source /tmp/build-source
python3 -m pip wheel --no-deps --no-build-isolation --no-index /tmp/build-source /tmp/build-source/src /tmp/build-source/fault39 -w /output/wheels
python3 -m pip install --no-deps --no-index --target /output/installed /output/wheels/*.whl
export PYTHONPATH=/output/installed
cd /tmp
python3 -B /source/experiments/c2c3_closure_20261117/installed_core_smoke.py --installed /output/installed --out /output/installed_smoke.json
unset PYTHONPATH
source /usr/local/Ascend/cann-8.5.1/set_env.sh
exec python3 -B /source/experiments/c2c3_closure_20261117/attention_cpu_preflight.py --repo /source --sdk /usr/local/Ascend/cann-8.5.1 --out /output/cann851/attention-preflight
'''
    command=['docker','create','--name','dcat-core-build-'+owner,
        '--label','dcat.cpu-build-owner='+owner,'--runtime=runc','--network=none',
        '--cpus=4','--memory=8g','-e','ASCEND_VISIBLE_DEVICES=void',
        '-e','PYTHONDONTWRITEBYTECODE=1','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0',
        '-e','LD_PRELOAD=','-e','ASCEND_CUSTOM_OPP_PATH=',
        '--mount','type=bind,src='+str(source)+',dst=/source,readonly',
        '--mount','type=bind,src='+str(out)+',dst=/output','--entrypoint','bash',IMAGE,'-c',script]
    receipt=dict(schema='demoncat.worker_core.device_free_build.v1',status='BUILDING',
        source_commit=manifest['base_commit'],source_manifest_sha256=sha(source/'source_manifest.json'),
        actual_npu_calls=0,image_id=IMAGE,command=command)
    def save():(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    container=subprocess.check_output(command,text=True,timeout=30).strip()
    receipt['container_id']=container
    def owned():
        item,=json.loads(subprocess.check_output(['docker','inspect',container],timeout=30))
        host=item['HostConfig']
        if (item['Id']!=container or item['Image']!=IMAGE
                or item['Config']['Labels'].get('dcat.cpu-build-owner')!=owner
                or host['Runtime']!='runc' or host['NetworkMode']!='none'
                or host['Privileged'] or host.get('Devices') or host.get('DeviceRequests')):
            raise RuntimeError('OWNED_DEVICE_FREE_CONTAINER_CHANGED')
        return item
    result=None
    try:
        receipt['devices']=owned()['HostConfig'].get('Devices') or []
        save()
        with (out/'build.log').open('xb') as log:
            result=subprocess.run(['docker','start','-a',container],stdout=log,stderr=subprocess.STDOUT,timeout=900)
        receipt.update(exit_code=result.returncode,status='BUILD_COMMAND_COMPLETED' if result.returncode==0 else 'BUILD_FAILED')
        if result.returncode==0:
            receipt['wheels']=[dict(name=p.name,sha256=sha(p)) for p in sorted((out/'wheels').glob('*.whl'))]
            receipt['installed_smoke_sha256']=sha(out/'installed_smoke.json')
            receipt['observer_receipt_sha256']=sha(out/'cann851/attention-preflight/result.json')
    except BaseException as error:
        receipt.update(status='BUILD_CONTROLLER_FAILED',error=repr(error))
        raise
    finally:
        if owned()['State']['Running']:
            subprocess.run(['docker','stop','--time','10',container],check=True,timeout=30,stdout=subprocess.DEVNULL)
        receipt['final_state']=owned()['State']
        subprocess.run(['docker','rm',container],check=True,timeout=30,stdout=subprocess.DEVNULL)
        receipt['owned_container_removed']=True
        save()
    print(json.dumps({k:receipt.get(k) for k in ('status','exit_code','actual_npu_calls','source_commit','observer_receipt_sha256')}))
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
