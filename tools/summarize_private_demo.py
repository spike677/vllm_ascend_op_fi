"""Offline new-user walkthrough/report from real TargetPlan and raw B/F/R.

Reuse EvaluationResult and the existing scorer/Dashboard. No generation, arm,
new thresholds, LLM judge or invented task/probability measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ascend_fi.inspect.dataset_effect import evaluate_dataset
from ascend_fi.evaluation.distortion.contracts import FaultConfigurationRecord
from ascend_fi.evaluation.distortion.report import write_report
from ascend_fi.contracts import EvaluationResult


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    load = lambda name: json.loads((a.evaluation / name).read_text(encoding='utf-8'))
    plan, identity, replay = load('target_plan.json'), load('identity.json'), load('replay_input.json')
    config = FaultConfigurationRecord.from_dict(replay['config'])
    result = evaluate_dataset(replay['suite'], config, replay['cycles'], evaluation_id=plan['plan_sha256'][:16])
    online = load('effect/evaluation_result.json')
    data = result.to_dict()
    # Preserve execution coverage limits; offline presentation must not promote
    # UNKNOWN/PARTIAL to a complete execution-window claim.
    for key in ('status', 'reason', 'model_targeting'):
        if key in online:
            data[key] = online[key]
    data['demo_reporting_revision'] = {
        'numeric_budget_source': 'ALL_ACCEPTED_CALLS_AGGREGATE_NOT_32_RETAINED_EVENTS',
        'online_raw_unchanged': True,
        'distortion_score_unchanged': data['distortion'] == online['distortion']}
    if not data['demo_reporting_revision']['distortion_score_unchanged']:
        raise ValueError('OFFLINE_SCORE_DIVERGED_FROM_LIVE; do not silently relabel')
    result = EvaluationResult.from_dict(data)
    write_report(a.out / 'effect', result, {'complete_cycles': 1, 'mode': 'OFFLINE_REAL_RAW_REPLAY'})
    lines = ['# Qwen3-0.6B：新用户识别、注入与效果评估演示', '',
        '本报告由真实 TargetPlan、IDENTITY 和 baseline/fault/recovery 原始记录生成。', '',
        '## 1. 识别到什么', '',
        f"- 实际目标：{len(plan['selected_modules'])} 个；角色：{', '.join(plan['roles'])}。",
        f"- 计划 SHA256：`{plan['plan_sha256']}`。",
        '- 根据实际模型类、配置、模块类和真实输入共同定位，不根据服务名猜架构。',
        '- 物理单卡只有 rank0；不存在第二个 worker，不能虚构 rank1 验证。', '',
        '| 目标角色 | 模块数 | 输入语义 | dtype | 局部特征维数 |', '| --- | ---: | --- | --- | --- |']
    targets = plan['workers'][0]['targets']
    for role in plan['roles']:
        ts = [t for t in targets if role in t['roles']]
        lines.append(f"| {role} | {len(ts)} | {', '.join(sorted({t['binding']['semantic_argument'] for t in ts}))} | "
            f"{', '.join(sorted({t['binding']['dtype'] for t in ts}))} | "
            f"{', '.join(map(str, sorted({t['binding']['feature_size'] for t in ts})))} |")
    lines += ['', '## 2. 注入的是哪一种量化误差', '',
        '模型权重保持 BF16；这不是给模型转换 W8A8，也不是修改原生量化 scale。',
        '在已识别的模块输入处，用原参考实现计算：', '',
        '```text', 'base_scale = absmax(input).to(FP32) / 127  # absmax 在原 dtype',
        f"fault_scale = max(base_scale, eps) × {plan['prospective_arm']['scale_multiplier']}",
        'q = clamp(round(input / fault_scale), -128, 127)',
        'faulty_input = (q × fault_scale).to(original_dtype)', '```', '',
        '倍率增大让量化步长变粗，可能增加归零和误差；不保证每个模型都产生病态输出。',
        f"每请求硬上限：{plan['request_window']['max_hits']}；公式：`{plan['request_window']['cap_formula']}`。",
        '每次请求独立 arm，finally disarm/remove，不在触顶时清零续打。', '',
        '## 3. 功能门禁', '',
        f"- IDENTITY：`{identity['validation']['status']}`；token pieces 精确一致：`{identity['identity_receipt']['token_pieces_exact']}`。",
        '- IDENTITY 是原样传递；倍率 1 仍然是有损量化，不用于代替无损校验。',
        '- 指标将命中、数值变化、输出差异和病态告警分开；恢复作为实验有效性门禁。', '',
        '## 4. 实测汇总', '',
        '| 样本 | B/F/R token 数 | hit | 改变元素比例 | 裁剪比例 | 首个 token-piece 分叉 | 规则告警 | 精确恢复 |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- | --- |']
    matrix = result.to_dict()['distortion']['cycles'][0]['matrix']
    by_id = {r['probe_id']: r for r in matrix}
    details, numerical = [], []
    for pid, pair in replay['cycles'][0]['pairs'].items():
        b, f, r = [pair[k] for k in ('baseline', 'fault', 'recovery')]
        worker = f['fault_transaction']['measurement_query']['results'][0]
        stats = list(worker['effect_summary']['per_module'].values())
        summed = lambda key: sum(s.get(key, 0) for s in stats)
        elements, changed, clips = [summed(k) for k in ('total_elements', 'changed_elements', 'clip_count')]
        nonzero, zeros = summed('sampled_nonzero_elements'), summed('sampled_new_zero_elements')
        insq, errsq = summed('sampled_input_sq_sum'), summed('sampled_error_sq_sum')
        row = by_id[pid]
        effect = row['effect']
        alarm = ', '.join(d['detector'] for d in effect['detectors'] if d['triggered'] is True) or '无'
        lines.append(f"| {pid} | {b['generation_length']}/{f['generation_length']}/{r['generation_length']} | "
            f"{worker['hit_count']} | {changed/elements:.2%} | {clips/elements:.2%} | {effect['first_divergence']} | {alarm} | {row['recovery_exact']} |")
        numeric = {'sample_id': pid, 'hit_count': worker['hit_count'], 'candidate_seen': worker['candidate_seen'],
            'reject_reasons': worker['reject_reasons'], 'changed_elements': changed, 'total_elements': elements,
            'changed_ratio': changed / elements, 'clip_rate': clips / elements,
            'fault_scale_min': min(s['fault_scale_min'] for s in stats),
            'fault_scale_max': max(s['fault_scale_max'] for s in stats),
            'sampled_new_zero_fraction_of_original_nonzero': zeros / nonzero if nonzero else None,
            'sampled_relative_l2': (errsq / insq) ** .5 if insq else None,
            'sampling_rule': worker['effect_summary']['dose_sampling'],
            'candidate_equals_hit': worker['candidate_seen'] == worker['hit_count'],
            'all_targets_hit': all(s['hit_count'] > 0 for s in stats) and len(stats) == len(targets),
            'source_request': f['source_ref']}
        numeric['derived_energy_metrics'] = data['numerical_effect']['per_probe'][pid]['aggregate']
        numerical.append(numeric)
        details += [f'### {pid}', '', f"原问题：{b['request']['messages'][0]['content']}", '',
            f"首次分叉位置（0 起始 token pieces）：{effect['first_divergence']}。整数 token IDs 未提供时保留 null。", '',
            f"数值采样：新增归零={numeric['sampled_new_zero_fraction_of_original_nonzero']:.2%}；相对 L2={numeric['sampled_relative_l2']:.6g}。",
            f"同采样能量派生：NMSE={numeric['derived_energy_metrics']['nmse']}；SQNR_dB={numeric['derived_energy_metrics']['sqnr_db']}（非独立证据）。",
            '采样为每模块第 1、65、129…次调用，不是全量元素统计。', '']
        for label, raw in [('正常输出', b), ('故障输出', f), ('恢复输出', r)]:
            details += [f'#### {label}', '', f"HTTP={raw['http_status']}；finish_reason={raw['finish_reason']}；tokens={raw['generation_length']}", '',
                '````text', raw['response_text'], '````', '']
    lines += ['', '## 5. 完整连续文本', '', *details, '## 6. 结论边界', '',
        '- 三道开放题、一个完整周期；不计算官方 MT-Bench 分数、不声称 Accuracy 下降或统计显著性。',
        '- 本轮没有 teacher-forced NLL/PPL；不能用 hidden-state 误差或生成文本冒充 PPL。',
        '- 目标调用 candidate/hit 一致不等于已有独立 scheduler 执行覆盖；full_request_window 保留 UNPROVEN。',
        '- length 截断、普通措辞变化、合法列表和表格，不单独视为严重病态。',
        '- 本次是四模块族组合且倍率 20，不能归因到某个模块族的独立贡献。',
        '- 原始 B/F/R、token pieces、逐模块聚合、计划与恢复记录保持可追溯；Dashboard 可离线查看。', '']
    (a.out / 'demo_walkthrough_zh.md').write_text('\n'.join(lines), encoding='utf-8')
    (a.out / 'numerical_effect.json').write_text(json.dumps(numerical, indent=2), encoding='utf-8')
    (a.out / 'replay_provenance.json').write_text(json.dumps({
        'source_file': str(a.evaluation / 'replay_input.json'),
        'source_sha256': hashlib.sha256((a.evaluation / 'replay_input.json').read_bytes()).hexdigest(),
        'no_npu': True, 'no_threshold_change': True}, indent=2), encoding='utf-8')
    print(str(a.out / 'demo_walkthrough_zh.md'))


if __name__ == '__main__':
    main()
