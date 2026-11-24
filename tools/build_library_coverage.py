"""Render the reviewed library-row ledger; identifiers are not public fault ids."""
import json
from pathlib import Path
from collections import Counter

PROGRESS_STATES = ('CLOSED_NO_INJECTION', 'AWAIT_FORMAL_TEST',
    'YELLOW_ADAPTER_READY_UNTESTED', 'REFERENCE_ADAPTER_PENDING',
    'HCCL_BITWISE_PENDING', 'FAULT33_EVIDENCE_PENDING', 'V4_D_PACKAGE_PENDING',
    'ADAPTER_READY_UNTESTED', 'REGISTERED')


def progress_counts(data):
    counts = Counter(r.get('progress_state') for r in data['rows'])
    if any(s not in PROGRESS_STATES for s in counts):
        raise ValueError('UNKNOWN_OR_MISSING_PROGRESS_STATE')
    return {s: counts[s] for s in PROGRESS_STATES if counts[s]}


def render(data):
    rows=data['rows']
    if [r['row'] for r in rows]!=list(range(1,42)):
        raise ValueError('EXACT_41_LIBRARY_ROWS_REQUIRED')
    groups={g:sum(r['category']==g for r in rows) for g in 'ABCDEF'}
    if groups!={'A':10,'B':17,'C':8,'D':1,'E':2,'F':3}:
        raise ValueError('LIBRARY_CATEGORY_COUNTS_CHANGED')
    head='''# 故障库41行覆盖台账（唯一当前口径）

2026-11-24修订。**库行号与公共故障编号是两个命名空间。** 库8是CP padding，不是公共8；库17是转置卷积padding，公共17仅提供相关空间误差现象。库27映射公共15，库41映射公共20，库39映射公共39mr。旧的12行映射已被本表替代，冻结发行物内历史文档不改写。

A公共交付10行；B黄区已有原语17行；C现象复现8行（新纳入30）；D仅V4 1行；E配置2行；F不适用3行。合计41行。黄区能力与状态仅引用回传，不把蓝区审查或示例当作黄区产品通过。

登记按“能力＋命名变体＋部署组合＋模式”一次记录，覆盖库行仅作备注。不得由同一行号自动推导公共机制。下列A类为已交付能力，不意味着本轮最终包已登记。

'''
    counts = progress_counts(data)
    progress = ('进度快照：'+data['progress_snapshot']+'。A–F 是归属，progress_state 是进度，两者独立。'
        '以下数字从逐行 JSON 汇总；52、56 行不在统计范围内。\n\n'
        '|进度状态|行数|库行|\n|---|---|---|\n')
    for state,count in counts.items():
        progress += f'|{state}|{count}|'+ '、'.join(str(r['row']) for r in rows if r['progress_state']==state)+'|\n'
    progress += ('\n历史公共证据不等于本轮通过。参考适配器通过 A2 后记 ADAPTER_READY_UNTESTED，'
        '标注等待黄区审查；正式测试审查并登记后才记 REGISTERED。第21行按区域寿命选择v1或v2，待黄区确认。\n\n'
        '|库行|类|进度|对应能力/变体|机理与范围|部署|证据或待补项|当前进度说明|\n|---|---|---|---|---|---|---|---|\n')
    return head+progress+''.join('|'+ '|'.join(str(r.get(k,'') if k=='progress_note' else r[k]).replace('|','／') for k in ('row','category','progress_state','capability','boundary','deployment','evidence','progress_note'))+'|\n' for r in rows)+data['notes']


def main():
    root=Path(__file__).resolve().parents[1]
    p=root/'docs/worker_core_v1/fault_library_coverage_20261124.json'
    p.with_suffix('.md').write_bytes(render(json.loads(p.read_bytes())).encode())


if __name__=='__main__':main()
