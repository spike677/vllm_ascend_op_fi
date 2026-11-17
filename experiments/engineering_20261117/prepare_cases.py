"""Generate the frozen first tensor-Provider NPU matrix, at most eight cases."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    rows=[]
    for mechanism in ('15.alias_detach','17.spatial_padding','18.api_half','8.restore_offset','22.wrong_row'):
        rows.append(dict(id='tensor'+mechanism.split('.')[0],image='quay.io/ascend/vllm-ascend:v0.18.0',timeout=600,
            mounts=[[a.source,'/source','ro']],
            env={'PYTHONPATH':'/source/src:/source/fault39/src:/source:/msprobe_packages',
                 'OMP_NUM_THREADS':'4'},
            command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec python3 /source/experiments/engineering_20261117/statistics_probe.py '
                    '--mechanism '+mechanism+' --out /evidence/result.json'))
    with a.out.open('x') as stream:json.dump(rows,stream,indent=2)


if __name__=='__main__':main()
