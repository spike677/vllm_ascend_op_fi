"""Bind scoped execution receipts to the 17 source-audit cards.

No NPU execution and no new acceptance inferred from a model family, a native
log or a historical PASS. Original audit/historical observations remain intact.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
BASE = 'bfdd6abc1adb5d08b034db7a96ec97ffbec0b9bf'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def main():
    audit = read(E/'audit_matrix.json')
    cpu = read(E/'cpu_final_receipt.json')
    receipts = read(E/'mechanism_receipts-a1.json')['cases']
    workers = read(E/'worker_receipts-a2.json')['cases']
    assert cpu['status'] == 'PASS_CPU_ONLY'
    total_tests = sum(r['tests'] for r in cpu['runs'])
    assert total_tests == 445
    summaries = {
        '8.restore_offset': ('AUDITED', '历史配对路由源码已找回；依赖和共享 Provider 未闭合，未在新 MoE 模型注入。'),
        '15.alias_detach': ('NATIVE_VERIFIED', '构造 NPU 模块 B/F/OFF 通过；视图写入正确，F 原 base 未更新，消费者读旧值。'),
        '17.spatial_padding': ('NATIVE_VERIFIED', '构造 NPU Conv3d B/F/OFF 通过；实际 padding 0,0,0→0,7,7，输出数值改变。'),
        '18.api_half': ('NATIVE_VERIFIED', '两个标准 RMSNorm 自动选点，8/8 适用调用返回减半，OFF 精确恢复。'),
        '20.buffer_substitute': ('CPU_VERIFIED', '共享 Worker 隔离/恢复通过；WeightNz 未观测，QMM TP2 每 rank 观测 3 次，但跨轮稳定目标未成立。'),
        '22.wrong_row': ('AUDITED', '历史错误行代理证据已核对，未找到可重用源码本体；不能称为 uint32 溢出。'),
        '25.reduce_scatter': ('CPU_VERIFIED', '补齐真实通信组/差值回放设计与 CPU 准入测试，尚无真实 collective 注入。'),
        '26.residual_stomp': ('CPU_VERIFIED', '保留历史推理闭环；NaN 位于写前目标统计，来源分配尚未证明，严格首差仍阻塞。'),
        '26.fia_destination': ('CPU_VERIFIED', '保留历史真实推理 writer/consumer 证据；共享 Worker 宿主适配待接入。'),
        'quant.batch_shared_scale': ('CPU_VERIFIED', '保留历史 TP2 decode；全张量统计不证明每请求 scale/Q 改变，逐行影响证据不足。'),
        '28.fa.legal_tiling': ('CPU_VERIFIED', 'StrategySession CPU 回归通过；共享模型 forward 的原生 FA Provider 接线尚缺。'),
        '28.fia.legal_tiling': ('CPU_VERIFIED', '保留合法切分历史数值结果；新 router NPU 未验收，不能承诺统计幅度达标。'),
        '28.fia.denominator_replay': ('CPU_VERIFIED', '保留分母重复累计独立机制及历史完整模型证据；新 router 尚缺匹配构建实测。'),
        '33.paged_decode': ('CPU_VERIFIED', '保留历史分页 decode；严格首差在后续统计，不能声称唯一定位 IFA，连续生成仍待测。'),
        '39.matmul_range': ('CPU_VERIFIED', '真实共享 Worker 的缺回放拒绝、隔离和恢复通过；缺可信 scheduler replay，未执行新注入。'),
        '39.output_proxy': ('AUDITED', '与 matmul_range 分开登记；历史 native 代理存在，专属共享 Worker 合同仍缺。'),
        '41.tail_column': ('AUDITED', '实际 dispatcher/cache writer/consumer 尚未闭合，保留拒绝与具体实施方案。'),
    }
    rows, coverage = [], []
    for old in audit['mechanisms']:
        row = deepcopy(old)
        mechanism = row['mechanism']
        level, conclusion = summaries[mechanism]
        row['historical_native'] = row.pop('native')
        row['historical_model_scope'] = row['model']['verified_scope']
        row['model']['current_scope'] = '六种实际 Transformers 小模型 CPU 结构/角色/透明前向；不是全部机制 NPU 验收'
        result = dict(level=level, conclusion=conclusion,
            actual_pretrained_model_fault_closed=False,
            common_worker_fault_injection_verified=False,
            current_npu_fault_gate9_closed=False,
            raw_statistics_published=False,
            historical_acceptance_inherited=False)
        row['cpu'] = dict(status='PASS_EXECUTABLE_CPU_REGRESSION', suite_receipt='cpu_final_receipt.json',
                          suite_total=total_tests, scope='Full suite; not 445 separate tests per mechanism')
        row['native'] = 'No new fault injection; historical scope explicitly separate'
        row['bf'] = 'No new independent B/F pair for this mechanism'
        row['current_level'] = level
        if mechanism.startswith(('15.', '17.', '18.')):
            number = mechanism.split('.')[0]
            passed = [r for r in receipts if r.get('fault') == mechanism and r['status'] == 'PASS_NATIVE_MECHANISM_ONLY']
            assert len(passed) == 2 and {r['arm'] for r in passed} == {'baseline','fault'}
            assert all(r['off_exact'] and r['repeat_stable'] and r['input_unchanged'] for r in passed)
            independent = read(E/f'summary{number}-a1.json')
            assert independent['status'] == 'INCOMPLETE_COMPARISON'
            assert all(s['status'] == 'ALIGNMENT_MISMATCH' for s in independent['steps'])
            result.update(scope='CONSTRUCTED_NPU_INFERENCE_MODULE',
                execution_receipts=[r['case'] for r in passed],
                native_receipt='mechanism_receipts-a1.json',
                independent_report=f'diagnostic{number}_summary.json',
                exact_off=True, steps=[0,1,2], warmup_forwards=1)
            row['gate'] = [g for g in row['gate'] if g != 'NEW_ROUTE_NPU_PENDING'] + ['G9_ALIGNMENT_MISMATCH']
            row['model']['current_scope'] = '构造的 eval/inference_mode NPU 模块；无预训练模型/Serving 闭环'
            row['cann'] = 'CANN 8.5.1 / vLLM-Ascend v0.18.0 image / torch-npu 2.9.0.post1+gitee7ba04'
            row['soc'] = '124 physical card5 / Ascend910B4 / A2'
            row['native'] = result['scope'] + ' B/F: PASS; ' + conclusion
            row['bf'] = dict(status='INCOMPLETE_COMPARISON', steps=[0,1,2],
                             first_difference=None, report=f'diagnostic{number}_summary.json')
            row['off'] = 'PASS exact after session exit; fresh initial state, not rollback of a damaged request'
            row['next_action'] = ('先补显式调用链对齐规则并独立审查；15 保留私有格式 null，寻找自然写入/消费者链；'
                                  '17/18 再接实际适用预训练模型，不扩大已验收范围。')
            row['coverage'] = dict(single_target='VERIFIED_CONSTRUCTED_MODULE_ONLY',
                multi_target='TWO_RMSNORM_MODULES_ONLY' if number=='18' else 'NOT_TESTED',
                every_match='VERIFIED_FOUR_FORWARDS_IN_THIS_MODULE_ONLY', rank_isolation='UNVERIFIED',
                resource_budget='small bounded module', group_recovery='OFF_OUTPUT_EXACT_IN_THIS_MODULE',
                common_worker_all_eligible='NOT_QUALIFIED')
        elif mechanism in ('20.buffer_substitute','39.matmul_range'):
            result.update(scope='REAL_VLLM_WORKER_QUARANTINE_RECOVERY_AND_READONLY_DISCOVERY',
                          execution_receipt='worker_receipts-a2.json',
                          trusted_replay_registered=False)
            row['native'] = conclusion
            row['off'] = 'Normal generation exact after refused register / unregister / detach; not fault OFF'
            row['gate'] += ['TRUSTED_REPLAY_BINDING_MISSING']
        elif mechanism == '26.residual_stomp':
            result['analysis_receipt'] = 'fault26_nan_analysis.json'
        elif mechanism in ('8.restore_offset','22.wrong_row'):
            result['analysis_receipt'] = 'legacy_source_audit.json'
        row['result'] = result
        rows.append(row)
        coverage.append(dict(mechanism=mechanism, scope='current sprint only',
                             qualification=row['coverage'], common_worker_all_eligible=False))
        card_path = ROOT/row['card']
        card = read(card_path)
        card['sprint_validation'] = result
        card['current_sprint_gaps'] = row['gate']
        card['coverage_current_sprint'] = row['coverage']
        card['current_next_action'] = row['next_action']
        # Existing gate fields concern the common Worker and stay unqualified.
        card['execution_gate_overlay'] = dict(scope=result.get('scope','CPU_OR_SOURCE_ONLY'),
            G6='MECHANISM_MODULE_PASS' if level=='NATIVE_VERIFIED' else 'NOT_NEWLY_ACCEPTED',
            G8='EXACT_MODULE_OFF' if level=='NATIVE_VERIFIED' else 'SEE_SCOPED_RECEIPT',
            G9='ALIGNMENT_MISMATCH' if level=='NATIVE_VERIFIED' else 'NOT_NEWLY_ACCEPTED')
        write(card_path, card)
        md = ROOT/'docs/overnight_fault_cards'/f'{mechanism}.md'
        text = md.read_text(encoding='utf-8').split('\n## 本轮执行回执')[0]
        md.write_text(text.rstrip()+f'\n\n## 本轮执行回执\n\n本轮等级：**{level}**。{conclusion}\n\n'
            '上文“当前缺口”是基线审计快照；执行后的缺口与历史状态分栏见'
            f'[最终17项结果矩阵](../../experiments/overnight_20261116/fault_matrix.json)及'
            f'[最终报告](../fault_closure_report.md)。本轮未授予共享 Worker 全量故障资格。\n', encoding='utf-8')
    counts = CounterLike(r['current_level'] for r in rows)
    write(E/'fault_matrix.json', dict(schema='demoncat.overnight.final_matrix.v1', baseline=BASE,
        mechanisms=rows, counts=counts, source_audited=17, cpu_tests=total_tests,
        new_npu_mechanism_pairs=3, new_pretrained_model_closed=0,
        new_common_worker_fault_closed=0, new_strict_independent_pairs_closed=0,
        scope_note='Historical MODEL_VERIFIED rows do not inherit current Worker acceptance'))
    write(E/'coverage_matrix.json', dict(schema='demoncat.overnight.coverage.v1', mechanisms=coverage))
    write(E/'native_receipts/index.json', dict(execution='../mechanism_receipts-a1.json',
        worker='../worker_receipts-a2.json', build='../fault20_bundle_review.json',
        cxx_ctypes_cpu_only='../worker_native_cpu_receipt.json', raw_data_root='/home/overnight-17-20261116'))
    write(E/'recovery_receipts/index.json', dict(module_cases=[
        {k:r[k] for k in ('case','fault','arm','off_exact','repeat_stable','input_unchanged','result')}
        for r in receipts if r['status']=='PASS_NATIVE_MECHANISM_ONLY'],
        worker_cases=[{k:r[k] for k in ('case','status','normal_generation_exact_after','recovery','detach','result')}
        for r in workers if r['status']=='PASS_QUARANTINE_RECOVERY'],
        note='Worker negative-gate recovery is not injected-fault OFF; module snapshot precedes close, OFF follows close'))
    write(E/'independent_reports/index.json', dict(policy=dict(rtol=1e-5,atol=1e-8),
        raw_statistics_published=False, global_first_difference_passes=0,
        reports=[f'../diagnostic{n}_summary.json' for n in (15,17,18)],
        diagnosis='../STATISTICS_ALIGNMENT_DIAGNOSIS.md',
        limitation='Matched local field differences do not override unmatched global call chains'))


def CounterLike(values):
    result = {}
    for value in values:
        result[value] = result.get(value,0)+1
    return result


if __name__ == '__main__':
    main()
