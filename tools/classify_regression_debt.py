"""Account for every historical failure occurrence, without claiming green tests."""
import argparse,collections,hashlib,json,re
from pathlib import Path

RETIRED=(
    'test_core_discovery.RegistryTests.test_attention_factories_are_distinct_and_require_actual_deployment',
    'test_core_discovery.RegistryTests.test_paged_s2_requires_exact_trusted_driver_and_preserves_stock_census',
)

def classify(review,run):
    ids=set(review['preexisting_shared_failure_ids']);rows=[]
    for kind in ('failures','errors'):
        for item in run[kind]:
            name=item['test']
            if name not in ids:continue
            codes=list(dict.fromkeys(re.findall(r'''(?:provider_code|code|reason)['"]\s*:\s*['"]([A-Z][A-Z0-9_:]+)''',item['tail'])))
            if not codes:
                codes=list(dict.fromkeys(re.findall(r'(?:RuntimeError|ValueError): ([A-Z][A-Z0-9_:]+)',item['tail'])))
            if any(name.startswith(old) for old in RETIRED):
                category='DEPRECATED';reason='旧研究Native/S2工厂经公共入口直达的断言已废弃；公共入口必须先准入，研究实现保留。'
                action='skip旧公共工厂选择用例；保留ordinary_admission/continuous Runtime负向准入覆盖。'
            elif 'ModuleNotFoundError' in item['tail']:
                category='ENVIRONMENT';reason='子进程cwd变化后相对PYTHONPATH失效；尚未进入机制断言。'
                action='修正子进程绝对import路径并复测；若暴露后续准入失败，重新归类。'
            else:
                category='REAL_DEFECT';reason=('当前记录出现拒绝代码 '+', '.join(codes)+'；需复测区分夹具契约与实现缺陷。' if codes else
                    '原回执未暴露可唯一归属的拒绝代码；保留为未解决测试/实现缺陷，不推定全部是夹具问题。')
                action='修复测试接线或拒绝契约，保留所有权/恢复负向断言；通过准入后暴露的实现问题继续修复。禁止批量skip。'
            filename=name.split('.')[0]+'.py'
            candidates=[Path('tests/worker_extension')/filename,Path('tests')/filename]
            test_file=next((p.as_posix() for p in candidates if p.exists()),None)
            rows.append(dict(occurrence=len(rows)+1,test=name,outcome=kind[:-1],category=category,
                defect_scope='TEST_OR_IMPLEMENTATION_PENDING_REEXECUTION' if category=='REAL_DEFECT' else None,
                reason=reason,action=action,test_file=test_file,observed_rejection_codes=codes,
                evidence_tail_sha256=hashlib.sha256(item['tail'].encode()).hexdigest(),resolved=False))
    if collections.Counter(r['outcome'] for r in rows)!=dict(failure=199,error=11):
        raise ValueError('HISTORICAL_OCCURRENCE_COUNT_CHANGED')
    return dict(schema='demoncat.regression_debt.v1',occurrences=len(rows),distinct_ids=len({r['test'] for r in rows}),
        categories=dict(collections.Counter(r['category'] for r in rows)),all_related_tests_passed=False,
        classification_is_not_test_execution=True,items=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('review');p.add_argument('run');p.add_argument('output');a=p.parse_args()
    result=classify(json.loads(Path(a.review).read_text()),json.loads(Path(a.run).read_text()))
    result['inputs']=[dict(file=Path(f).name,sha256=hashlib.sha256(Path(f).read_bytes()).hexdigest()) for f in (a.review,a.run)]
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
