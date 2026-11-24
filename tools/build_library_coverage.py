"""Render the reviewed library-row ledger; identifiers are not public fault ids."""
import json
from pathlib import Path


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

|库行|类|对应能力/变体|机理与范围|部署|证据或待补项|
|---|---|---|---|---|---|
'''
    return head+''.join('|'+ '|'.join(str(r[k]).replace('|','／') for k in ('row','category','capability','boundary','deployment','evidence'))+'|\n' for r in rows)+data['notes']


def main():
    root=Path(__file__).resolve().parents[1]
    p=root/'docs/worker_core_v1/fault_library_coverage_20261124.json'
    p.with_suffix('.md').write_bytes(render(json.loads(p.read_bytes())).encode())


if __name__=='__main__':main()
