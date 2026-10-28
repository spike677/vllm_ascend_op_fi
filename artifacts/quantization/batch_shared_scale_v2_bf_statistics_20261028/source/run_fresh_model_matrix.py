"""One fresh engine/process per observation: cache addresses and slots align."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
outputs=root/'outputs/fresh_matrix'
outputs.mkdir(parents=True,exist_ok=False)
source=outputs/'source'
source.mkdir()
source_names=('model_probe.py','model_hook.py','backend_control.py','sitecustomize.py',
              'native_backend.cpp','build_backend.py','run_fresh_model_matrix.py')
manifest={}
for name in source_names:
    data=(root/name).read_bytes()
    (source/name).write_bytes(data)
    manifest[name]=hashlib.sha256(data).hexdigest()
(source/'SHA256.json').write_text(json.dumps(manifest,indent=2))
records=[]
sequence=0
for condition in ('letter_counts_B2','even_count_B4'):
    native=None
    observed=None
    stages=[('B',False,False),('B',False,True),('H',True,True),('H1',True,True),
            ('F',True,True),('F2',True,True),('R',True,True)]
    if condition=='letter_counts_B2': stages += [('X',True,True),('X2',True,True),('XR',True,True)]
    for stage,backend,observe in stages:
        ident=f'run{sequence:03d}'
        destination=outputs/ident
        command=[sys.executable,str(root/'model_probe.py'),'--output',str(destination),
                 '--condition',condition,'--single-stage',stage,'--max-tokens','16']
        if observe: command+=['--observe']
        if backend: command+=['--backend']
        reference=observed if backend else native
        if reference: command+=['--reference',str(reference/'result.json')]
        if stage in ('F','F2','X','X2'): command+=['--allow-fault']
        start=time.time()
        with (outputs/(ident+'.log')).open('x') as stream:
            process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=480)
        record={'id':ident,'condition':condition,'stage':stage,'backend':backend,'observe':observe,
                'return':process.returncode,'seconds':time.time()-start}
        records.append(record)
        (outputs/'execution.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(record),flush=True)
        if process.returncode: raise RuntimeError('Fresh engine run failed: '+ident)
        result=json.loads((destination/'result.json').read_text())
        if result['status']!='completed' or len(result['runs'])!=1: raise RuntimeError('Incomplete single-run record')
        if not backend:
            if observe: observed=destination
            else: native=destination
        sequence+=1
(outputs/'completed.json').write_text(json.dumps({'status':'EXECUTED','runs':len(records),
    'scope':'16 generated tokens maximum: validates the first decode fault and short propagation, not final-answer quality'}))
