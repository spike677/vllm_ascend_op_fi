"""Fresh CPU-only copy of the audited runtime/vendor/inputs, not old outputs."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

OLD=Path('/home/fault28-model-coverage-20261030')
ROOT=Path('/home/fault28-m-only-model-20261109')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not ROOT.exists()
    assert sha(OLD/'runtime_manifest.json')=='ca0bf188bdc031ea3e4249c03ac13a9a8f751cd5ae8485655d9048c9f6cdba3b'
    assert sha(OLD/'native_vendor_stage_a1/vendor_manifest.json')=='42173485d5d3c86b57cd8a847dbce2dfd1fc42004c1a451bb282b7aa211d9ef1'
    ROOT.mkdir()
    for name in ('runtime','inputs','native_vendor_stage_a1'):
        shutil.copytree(OLD/name,ROOT/name,symlinks=True)
    for name in ('runtime_manifest.json','model_manifest.json','tokenizer_manifest.json',
                 'source_fingerprints.json','memory_api_source_audit.json'):
        shutil.copyfile(OLD/name,ROOT/name)
    expected=json.loads((ROOT/'runtime_manifest.json').read_bytes())['files']
    assert set(expected)=={p.relative_to(ROOT/'runtime').as_posix() for p in (ROOT/'runtime').rglob('*') if p.is_file()}
    for name,row in expected.items():assert sha(ROOT/'runtime'/name)==row['sha256']
    devices=[]
    for identity in subprocess.check_output(['docker','ps','-q','--no-trunc'],text=True).split():
        raw=json.loads(subprocess.check_output(['docker','inspect',identity],text=True))[0]
        host=raw['HostConfig']
        devices.append({'id':identity,'name':raw['Name'].lstrip('/'),'image':raw['Image'],
                        'command':raw['Config'].get('Cmd'),'privileged':host.get('Privileged'),
                        'devices':host.get('Devices'),'device_requests':host.get('DeviceRequests'),
                        'mounts':[{'source':m.get('Source'),'destination':m['Destination'],'rw':m['RW']}
                                  for m in raw.get('Mounts',[])]})
    receipt={'schema':'fault28.m_only.cpu_preparation.v1','npu_executed':False,
             'source_root':str(OLD),'new_root':str(ROOT),'runtime_manifest_sha256':sha(ROOT/'runtime_manifest.json'),
             'vendor_manifest_sha256':sha(ROOT/'native_vendor_stage_a1/vendor_manifest.json'),
             'input_manifest_sha256':sha(ROOT/'inputs/manifest.json'),'containers':devices,
             'free_disk_bytes':shutil.disk_usage(ROOT).free}
    (ROOT/'analysis').mkdir()
    (ROOT/'analysis/cpu_preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('new_root','runtime_manifest_sha256','vendor_manifest_sha256',
                                           'input_manifest_sha256','free_disk_bytes','npu_executed')}))


if __name__=='__main__':main()
