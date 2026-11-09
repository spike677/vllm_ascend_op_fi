"""Create a new immutable offline review input for two clean controller attempts."""
import argparse
import json
from pathlib import Path

ROOT=Path('/home/fault28-m-only-model-20261109')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--b1-attempt',required=True);p.add_argument('--b8-attempt',required=True)
    p.add_argument('--b1-analysis',required=True);p.add_argument('--b8-analysis',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cases=[]
    for batch,attempt_name,analysis_name in ((1,a.b1_attempt,a.b1_analysis),(8,a.b8_attempt,a.b8_analysis)):
        assert '/' not in attempt_name and '/' not in analysis_name
        attempt=ROOT/'outputs'/attempt_name;analysis=ROOT/'outputs'/analysis_name;name=f'main_b{batch}'
        assert json.loads((attempt/'completion.json').read_bytes())['status']=='CAPTURE_COMPLETED'
        for path in (analysis/'framework.json',analysis/'native_census.json',analysis/'statistics/summary.json'):
            assert path.is_file()
        cases.append({'case_dir':str(attempt/name),'framework_report':str(analysis/'framework.json'),
                      'native_census_report':str(analysis/'native_census.json'),
                      'statistics_report':str(analysis/'statistics/summary.json'),
                      'completion':str(attempt/'completion.json'),'log':str(attempt/(name+'.log'))})
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'suite.json').open('x') as stream:
        json.dump({'schema':'nes.model_coverage.acceptance_inputs.v1','cases':cases},stream,indent=2);stream.write('\n')
    print(json.dumps({'status':'REVIEW_INPUTS_WRITTEN','path':str(a.output/'suite.json')}))


if __name__=='__main__':main()
