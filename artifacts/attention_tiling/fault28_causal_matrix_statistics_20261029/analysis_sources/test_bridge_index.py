"""Bounded CPU tests of four-BASE selection and read-only attempt-index planning.

Synthetic controller documents below test indexing only. They are temporary and
are never presented as native capture, public raw, or numerical evidence.
"""
import copy
import json
from pathlib import Path
import tempfile

import index_attempts
import profile_evidence as pe
from verify_base_bridge import select_base_cells


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    here=Path(__file__).resolve().parent;root=here.parents[2]
    real=json.loads((here.parent/'setup/primary/manifest.json').read_bytes())
    checks=[]
    assert len(real['cases'])==16 and len(select_base_cells(real))==4
    checks.append({'name':'real_frozen16_selects_exact4_BASE_without_other_results','status':'PASS'})
    def rejects(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):checks.append({'name':name,'status':'PASS_REJECTED'});return
        raise AssertionError('Unexpected acceptance: '+name)
    wrong=copy.deepcopy(real);wrong['cases'].pop(0)
    rejects('missing_BASE_repeat',lambda:select_base_cells(wrong))
    wrong=copy.deepcopy(real);wrong['cases'][1]=copy.deepcopy(wrong['cases'][0])
    rejects('duplicate_BASE_cell',lambda:select_base_cells(wrong))
    with tempfile.TemporaryDirectory(prefix='_f28index_',dir=root) as temp:
        base=Path(temp).resolve();assert base.parent==root and base.is_relative_to(root)
        matrix_file=base/'setup/manifest.json';rows=[];configs={}
        for i in range(2):
            name=f'fixture_b1_r{i}';file=base/'setup'/(name+'.json')
            write(file,{'case_id':name,'profile':'base','repeat_index':i,'batch_size':1})
            row={'case_id':name,'profile':'base','repeat_index':i,'batch_size':1,'file':file.name,'sha256':pe.sha(file)}
            rows.append(row);configs['setup/'+file.name]=file.read_bytes()
        write(matrix_file,{'schema':'fault28.causal.matrix.v1','phase':'primary','cases':rows})
        source={key:pe.digest(value) if hasattr(pe,'digest') else __import__('hashlib').sha256(value).hexdigest() for key,value in configs.items()}
        attempts=[]
        for i,row in enumerate(rows):
            attempt=base/f'attempt{i}';attempts.append(attempt)
            result=attempt/row['case_id']/'result.json'
            write(result,{'status':'CAPTURE_COMPLETED','case_id':row['case_id'],'source_case_sha256':row['sha256']})
            write(attempt/'completion.json',{'status':'CAPTURE_COMPLETED','cleanup_errors':[],
                'owned_container_removed':True,'device_lock_released':True,'cases':[{
                    'case_id':row['case_id'],'status':'CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE','rc':0,'result_sha256':pe.sha(result)}]})
            write(attempt/'source_manifest.json',source)
            write(attempt/'selected_cases.json',[{'case_id':row['case_id'],'file':'setup/'+row['file']}])
            for relative,payload in configs.items():
                dest=attempt/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
            (attempt/(row['case_id']+'.log')).write_text('SYNTHETIC INDEX FIXTURE ONLY\n',encoding='utf-8')
            write(attempt/(row['case_id']+'.controller.json'),{'rc':0})
        before={str(p):pe.sha(p) for p in base.rglob('*') if p.is_file()}
        planned=index_attempts.plan(matrix_file,attempts)
        assert len(planned['cases'])==2 and len(planned['links'])==6
        assert before=={str(p):pe.sha(p) for p in base.rglob('*') if p.is_file()}
        checks.append({'name':'two_attempts_planned_without_original_changes_or_symlink_creation','status':'PASS'})
        rejects('missing_attempt_case',lambda:index_attempts.plan(matrix_file,attempts[:1]))
        rejects('duplicate_case_attempt',lambda:index_attempts.plan(matrix_file,attempts+[attempts[0]]))
        p=attempts[0]/'completion.json';saved=p.read_bytes();d=json.loads(saved);d['owned_container_removed']=False;write(p,d)
        rejects('unclean_attempt',lambda:index_attempts.plan(matrix_file,attempts));p.write_bytes(saved)
        p=attempts[0]/rows[0]['case_id']/'result.json';saved=p.read_bytes();p.write_bytes(saved+b' ')
        rejects('result_changed_after_completion',lambda:index_attempts.plan(matrix_file,attempts));p.write_bytes(saved)
        p=attempts[1]/'source_manifest.json';saved=p.read_bytes();p.write_bytes(saved+b' ')
        rejects('different_source_manifest_bytes',lambda:index_attempts.plan(matrix_file,attempts));p.write_bytes(saved)
    report={'status':'PASS_CPU_BASE_SELECTION_AND_INDEX_PLAN','npu_api_calls':0,'checks':checks,
            'native_bridge_execution_tested':False,'linux_symlink_materialization_tested':False}
    (here/'bridge_index_cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'checks':len(checks)}))


if __name__=='__main__':main()
