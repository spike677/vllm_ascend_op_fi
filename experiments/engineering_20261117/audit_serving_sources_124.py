"""Read installed Attention implementations in device-free containers."""
import argparse
import json
from pathlib import Path
import subprocess

CODE = '''import hashlib, importlib.util, importlib.metadata, json
from pathlib import Path
out=Path('/evidence')
record={'actual_npu_calls':0,'modules':{},'source_files':{}}
for name in ('vllm','vllm_ascend'):
    spec=importlib.util.find_spec(name)
    record['modules'][name]=str(spec.origin) if spec else None
    if spec:
        root=Path(spec.origin).parent
        paths=list((root/'attention').rglob('*.py')) if name=='vllm_ascend' else []
        paths += list((root/'quantization/methods').glob('w8a8_dynamic.py')) if name=='vllm_ascend' else []
        if name=='vllm':
            paths += list((root/'model_executor/layers/attention').rglob('*.py'))
            paths += list((root/'attention').glob('layer.py'))
            paths += list(root.glob('forward_context.py'))
        for path in sorted(set(paths)):
            raw=path.read_bytes();relative=path.relative_to(root)
            if len(raw)>2*1024*1024:raise RuntimeError('UNBOUNDED_SOURCE')
            dest=out/name/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(raw)
            record['source_files'][name+'/'+str(relative)]={'source':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
(out/'inventory.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--image',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    cmd=['docker','run','--rm','--network','none','--runtime','runc','-e','ASCEND_VISIBLE_DEVICES=void',
         '--mount','type=bind,src='+str(a.out.resolve())+',dst=/evidence',
         '--entrypoint','python3',a.image,'-c',CODE]
    with (a.out/'process.log').open('xb') as stream:result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=90)
    print(json.dumps(dict(image=a.image,exit_code=result.returncode,out=str(a.out),actual_npu_calls=0)))
    return result.returncode

if __name__=='__main__':raise SystemExit(main())
