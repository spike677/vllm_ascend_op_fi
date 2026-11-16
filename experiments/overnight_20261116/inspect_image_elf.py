"""CPU-only ELF inventory in owned containers; no devices and no library loading."""
import argparse
import json
from pathlib import Path
import subprocess
import uuid

IMAGES = {
    '851': ('quay.io/ascend/vllm-ascend:v0.18.0', '8.5.1'),
    '90': ('cann:9.0.0-910b-torch2.10.0_npu-atk26.4.30', '9.0.0'),
    '91': ('cann:9.1.0-910b-torch2.10.0_npu-atk26.7.8', '9.1.0'),
    '92': ('cann:9.2.0-910b-torch2.10.0_npu-atk26.7.8', '9.2.0'),
}
CODE = r'''
import hashlib,json,re,subprocess,sys
from pathlib import Path
version=sys.argv[1];root=Path('/usr/local/Ascend/cann-'+version+'/aarch64-linux')
rows=[]
for name in ('libopapi.so','libopapi_nn.so','libnnopbase.so','libascendcl.so'):
 p=root/'lib64'/name;raw=p.read_bytes()
 notes=subprocess.run(['readelf','-n',str(p)],check=True,text=True,capture_output=True).stdout
 dynamic=subprocess.run(['readelf','-d',str(p)],check=True,text=True,capture_output=True).stdout
 build=re.search(r'Build ID:\s*([0-9a-fA-F]+)',notes)
 rows.append(dict(name=name,path=str(p.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
   elf_build_id=build.group(1) if build else None,needed=re.findall(r'\(NEEDED\).*?\[(.*?)\]',dynamic),
   soname=re.findall(r'\(SONAME\).*?\[(.*?)\]',dynamic),
   notes_sha256=hashlib.sha256(notes.encode()).hexdigest(),dynamic_sha256=hashlib.sha256(dynamic.encode()).hexdigest()))
print(json.dumps(dict(version=version,libraries=rows,actual_npu_calls=0,libraries_loaded=False)))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise RuntimeError('Use a fresh output path')
    rows=[]
    for tag,(image,version) in IMAGES.items():
        nonce=uuid.uuid4().hex;name='overnight-env-'+tag+'-'+nonce[:10]
        command=['docker','run','--name',name,'--runtime=runc','--network','none','--cpus','2',
                 '--memory','4g','--pids-limit','128','--label','dcat.task=overnight-environment-20261116',
                 '--label','dcat.nonce='+nonce,'-e','ASCEND_VISIBLE_DEVICES=void','-e','LD_PRELOAD=',
                 '--mount','type=bind,src=/dev/null,dst=/etc/ld.so.preload,readonly',
                 '--entrypoint','python3',image,'-c',CODE,version]
        result=None
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=180)
            info=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
            if info['HostConfig'].get('Devices') or info['HostConfig'].get('Privileged'):
                raise RuntimeError('Unexpected device permission')
            row=dict(tag=tag,image=image,image_id=info['Image'],container_id=info['Id'],
                     devices=info['HostConfig'].get('Devices'),runtime=info['HostConfig']['Runtime'],
                     returncode=result.returncode,stderr=result.stderr,
                     data=json.loads(result.stdout) if result.returncode==0 else None)
            rows.append(row)
        finally:
            query=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
            if query.returncode==0:
                info=json.loads(query.stdout)[0]
                if info['Config'].get('Labels',{}).get('dcat.nonce')!=nonce:
                    raise RuntimeError('Container ownership changed')
                if info['State']['Running']:
                    subprocess.run(['docker','stop','--time','5',info['Id']],check=True,capture_output=True)
                subprocess.run(['docker','rm',info['Id']],check=True,capture_output=True)
    with a.output.open('x') as stream:
        json.dump(dict(schema='demoncat.overnight.elf_inventory.v1',rows=rows,actual_npu_calls=0,
                       libraries_loaded=False,all_owned_containers_removed=True),stream,indent=2)
    print(json.dumps(dict(status='PASS_CPU_ELF_ONLY' if all(r['returncode']==0 for r in rows) else 'FAILED',
                          rows=len(rows),output=str(a.output))))
    return int(any(r['returncode']!=0 for r in rows))


if __name__=='__main__':raise SystemExit(main())
