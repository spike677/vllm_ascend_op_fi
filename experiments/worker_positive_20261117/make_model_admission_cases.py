import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--base',required=True)
parser.add_argument('--out',required=True);parser.add_argument('--model',choices=['deepseek','glm'],required=True)
args=parser.parse_args();root=Path('/home/worker-positive-20261117')
if any(not n.replace('-','').replace('.','').isalnum() for n in (args.base,args.out)):
    raise ValueError('Invalid label')
case=json.loads((root/args.base).read_text())[0]
case['id']='pretrained-admission-'+args.model
for mount in case['mounts']:
    if mount[1]=='/model':
        mount[0]=('/home/haoran/models/DeepSeek-V2-Lite-W8A8' if args.model=='deepseek' else
                  str(root/'models/GLM-4-9B-0414'))
    elif mount[1]=='/probe.py':mount[0]=str(root/'model_admission_probe.py')
case['command']=case['command'].replace(' --statistics','')
with (root/args.out).open('x') as stream:json.dump([case],stream,indent=2)
