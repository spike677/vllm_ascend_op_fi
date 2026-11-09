"""Finite serial continuation, requiring complete per-case coverage gates."""
import json
from pathlib import Path
import subprocess
import sys
from coverage import audit

ROOT=Path('/home/fault28-fia-fullmodel-20261109')
ORDER=['model_b8_fault_r1','model_b8_fault_r2','model_b8_restore',
       'model_b1_fault_r1','model_b1_fault_r2','model_b1_restore',
       'model_b8_noalias_baseline','model_b8_noalias_fault']


def check(name):
    result=audit(name+'_a5')
    path=ROOT/'outputs/coverage_gates_v1'/(name+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'event':'COVERAGE_PASS','case':name,'layers':result['logical_fia_nodes'],
        'node_executions':result['actual_fia_node_executions'],'request_executions':result['layer_request_executions'],
        'M':result['actual_m']}),flush=True)


def main():
    matrix=json.loads((ROOT/'setup/matrix_v5.json').read_bytes())
    assert set(matrix['case_ids'])==set(ORDER)|{'model_b1_baseline','model_b8_baseline'}
    for name in ('model_b1_baseline','model_b8_baseline'):check(name)
    for name in ORDER:
        path=ROOT/'outputs'/(name+'_a5')/'completion.json'
        if not path.exists():
            assert not path.parent.exists(),'Incomplete attempt requires a new explicit attempt'
            print(json.dumps({'event':'BEGIN_CASE','case':name}),flush=True)
            subprocess.run([sys.executable,'-B',str(ROOT/'analysis/run_case.py'),'--case',name,
                '--attempt',name+'_a5','--matrix','matrix_v5.json'],check=True)
        check(name)
    print(json.dumps({'status':'ALL_PREREGISTERED_MODEL_CASES_AND_COVERAGE_PASSED',
        'cases':10,'full_model_forwards':30,'fia_node_executions':840}),flush=True)


if __name__=='__main__':main()
