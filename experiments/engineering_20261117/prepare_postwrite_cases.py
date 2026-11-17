import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--native',required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hf-model',help='Add a separate pretrained Qwen/HF 39 B/F pair')
    p.add_argument('--hf-only',action='store_true',help='Run only the separate pretrained pair')
    a=p.parse_args();cases=[]
    if a.hf_only and not a.hf_model:p.error('--hf-only requires --hf-model')
    for mechanism in ('39.output_proxy','41.tail_column'):
        for arm in ('baseline','fault'):
            cases.append(dict(id=mechanism.split('.')[0]+'-'+arm,
                image='quay.io/ascend/vllm-ascend:v0.18.0',timeout=900,
                mounts=[[a.source,'/source','ro'],[a.native,'/native','ro']],
                env={'PYTHONPATH':'/source/src:/source/fault39/src:/source:/msprobe_packages','OMP_NUM_THREADS':'4'},
                command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
                    'exec python3 /source/experiments/engineering_20261117/postwrite_probe.py --mechanism '+mechanism+
                    ' --arm '+arm+' --library /native/libdcat_postwrite.so --sha256 '+a.sha256+' --out /evidence --statistics'))
    if a.hf_model:
        import copy
        for case in cases[:2]:
            item=copy.deepcopy(case);item['id']=item['id'].replace('39-','39hf-')
            item['mounts'].append([a.hf_model,'/model','ro'])
            item['command']+=' --model /model';cases.append(item)
    if a.hf_only:cases=[item for item in cases if item['id'].startswith('39hf-')]
    with a.out.open('x') as stream:json.dump(cases,stream,indent=2)


if __name__=='__main__':main()
