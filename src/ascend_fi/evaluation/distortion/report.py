"""Standalone escaped HTML matrix and machine-readable effect artifacts."""

from __future__ import annotations

import html
import json
from pathlib import Path


def _pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _dataset_configuration(config):
    """Describe the recorded configuration, never a frozen R1 default."""
    target = (f"rank {config.get('targeted_ranks')}；"
              f"{len(config.get('targeted_layers', []))}/{config.get('available_layers')} 层；"
              f"模块族 {', '.join(config.get('module_families', [])) or 'UNKNOWN'}；"
              f"{len(config.get('actual_module_names', []))} 个明确目标。"
              f"{config.get('strategy')}；strength={config.get('strength')}。")
    policy = (f"Hit policy：{config.get('hit_policy')}。"
              "安全上限不等于剂量；实际覆盖、预算触顶和拒绝见逐请求证据。"
              "仅 hook 命中比例不证明完整执行步覆盖。")
    return target, policy


def render_dashboard(result, *, mode="distortion"):
    data = result.to_dict()
    if 'single_request_effect' in data:
        return _single_dashboard(data)
    d, config = data["distortion"], data["fault_configuration"]
    # Preserve raw newline/trailing-space evidence in the DOM without placing
    # evidence whitespace at physical Git line endings in the generated HTML.
    esc = lambda v: html.escape(str(v), quote=True).replace("\r", "&#13;").replace("\n", "&#10;")
    n, p = d["total_cycles"], d["probe_count"]
    dataset = data.get('dataset_delivery')
    cards = f"""<section class="cards"><article><h2>HOW MUCH DID WE INJECT?</h2>
<p>故障配置 · 不等于退化严重度</p><dl>
<dt>Capability</dt><dd>{esc(config['capability'])}</dd>
<dt>Module families</dt><dd>{esc(', '.join(config['module_families']) or 'UNKNOWN')}</dd>
<dt>Layers / ranks</dt><dd>{len(config['targeted_layers'])} / {esc(config['available_layers'])} 层 · rank {esc(config['targeted_ranks'])} / TP{esc(config['world_size'])}</dd>
<dt>Strength</dt><dd>{esc(config['strength'])}</dd><dt>Hit policy</dt><dd>{esc(config['hit_policy'])}</dd>
<dt>Mechanism</dt><dd>{esc(d['mechanism_gate'])} · {d['mechanism_valid_requests']}/{d['mechanism_requests']} requests</dd></dl></article>
<article><h2>WHAT DID USERS ACTUALLY SEE?</h2><p>固定输入下的可见变化</p>
<div class="headline">{len(d['stable_distorted_probes'])} / {p}<span>stable distorted probes</span></div>
<p>{d['complete_cycles']}/{n} 个周期完整；稳定性要求同一道题，而非不同题数量相同。</p>
<dl><dt>Stable families</dt><dd>{esc({k: len(v) for k,v in d['stable_phenotype_families'].items()})}</dd>
<dt>Crash</dt><dd>{d['crash_count']}</dd><dt>失常样本恢复</dt><dd>{d['recovery']['restored_pairs']} / {d['recovery']['fault_induced_pairs']} pairs · {esc(d['recovery'].get('phenotype_recovery_gate','UNKNOWN'))}</dd>
<dt>全量恢复 Gate</dt><dd>{d['recovery']['gate']}（含全用例任务状态与 runtime，不用局部比例代替）</dd></dl>
<p class="badge">{esc(d['robust_gate'])}</p></article></section>"""
    if dataset:
        target, policy = _dataset_configuration(config)
        cards = f'''<section class="cards"><article><h2>正式数据 → 真实目标 → 本请求变化</h2>
<p>{esc(dataset['dataset']['repository'])} · revision {esc(dataset['dataset']['revision'])}</p>
<p>首轮问题：{esc(dataset['question_ids'])}；未翻译或改写；不是官方 MT-Bench 评分。</p>
<p>{esc(target)}</p>
<p>实际 hit：{d['measurement_hits']}；changed elements：{d['changed_elements']}。</p>
<p>{esc(policy)}</p>
<details><summary>逐请求 rank / layers / hit budget / reject 证据</summary><pre>{esc(_pretty(dataset['budget_evidence']))}</pre></details></article>
<article><h2>有损集成与可见异常：两个独立结论</h2>
<p>DATASET_BACKED_LOSSY_INTEGRATION：{esc(dataset['DATASET_BACKED_LOSSY_INTEGRATION'])}</p>
<p>VISIBLE_DISTORTION_OBSERVATION（规则层）：{esc(dataset['VISIBLE_DISTORTION_OBSERVATION'])}</p>
<p>轨迹变化：{dataset['response_divergence']}/{p}；规则告警：{len(dataset['visible_sample_ids'])}/{p}。</p>
<p>异常样本：{esc(dataset['visible_sample_ids'])}；恢复全文精确一致：{dataset['exact_recovery']}/{p}。</p>
<p>Mechanism {esc(d['mechanism_gate'])}；Recovery {esc(d['recovery']['gate'])}；Crash {d['crash_count']}。</p>
<p>1 个周期，不证明跨周期稳定；旧 robust gate：NOT_APPLICABLE。</p>
<p>开放式回答未打任务分；SEMANTIC_REVIEW_REQUIRED。输出分叉不等于胡言乱语。</p></article></section>'''
    context_note = ''
    if data.get('model_targeting'):
        targeting = data['model_targeting']
        window = targeting.get('window', {})
        context_note += ('<article><h2>自动目标计划：机制与完整窗口分别验收</h2><pre>'
                         + esc(_pretty({'plan_sha256': targeting.get('plan_sha256'),
                             'status': targeting.get('status', data.get('status')),
                             'mechanism': window.get('mechanism'),
                             'target_call_window': window.get('target_call_window'),
                             'full_request_window': window.get('full_request_window'),
                             'recovery_exact': targeting.get('recovery_exact')})) + '</pre></article>')
    if data.get('rule_context_review'):
        context_note += '<article><h2>告警上下文：保留规则分数，不冒充语义结论</h2><p>标点告警命中了 Markdown 表格分隔线，不能仅凭该规则认定真实刷屏；语义仍需独立复核。</p><details><summary>分隔线原文与字符位置</summary><pre>' + esc(_pretty(data['rule_context_review'])) + '</pre></details></article>'
    rows = []
    for cycle in d["cycles"]:
        for row in cycle["matrix"]:
            original_prompt = ''
            if dataset:
                matching = [s for s in dataset.get('samples', [])
                            if row['probe_id'] == f"mtbench_{s['question_id']}_turn{s['turn_index']}"]
                if matching:
                    original_prompt = '<details open><summary>官方原始输入</summary><pre>' + esc(matching[0]['prompt']) + '</pre></details>'
            families = row["effect"]["families"]
            tags = ("distorted " if row["visible_distortion"] else "") + " ".join(families).lower()
            if row["effect"]["verdict"] == "CRASH":
                tags += " crash"
            if row["recovery_restored"] is False:
                tags += " recovery_failure"
            cells = []
            for phase in ("baseline", "fault", "recovery"):
                evidence = row[phase]
                comparison = row['effect'] if phase != 'recovery' else row['recovery_effect']
                verdict = comparison.get('display_status', '未检测到已覆盖的异常') if phase != 'baseline' else '正常对照（不等于已验证语义正确）'
                text = evidence.get('full_text') if dataset else evidence["text"]
                text = text if text is not None else "UNKNOWN: raw output missing"
                reason = "; ".join(x["reason"] for x in row["effect"]["detectors"] if x["triggered"] is True) if phase == "fault" else ""
                task = row.get('task_outcomes', {}).get(phase, {})
                repeated = next((x.get('baseline_value' if phase == 'baseline' else 'fault_value') for x in comparison['detectors'] if x['detector'] == 'repetition'), None) or {}
                details = {'task_outcome': task, 'repeated_fragment': repeated.get('repeated_fragment'),
                           'semantic_incoherence': 'NOT_PROVEN', 'source_ref': evidence.get('source_ref')}
                trimmed = '是，仅缩短展示，非模型停止位置' if evidence.get('truncated_excerpt') and not dataset else '否'
                facts = (f"任务：{task.get('status','NOT_EVALUATED')}；终止：{evidence.get('finish_reason')}；"
                         f"原生成 {evidence.get('generation_length')} tokens / {evidence.get('max_tokens')} 预算；"
                         f"全文 {evidence.get('character_count')} 字符；展示截短：{trimmed}。")
                cells.append(f'<td><span class="badge">{esc(verdict or "UNKNOWN")}</span><p class="generation-facts">{esc(facts)}</p><pre>{esc(text)}</pre><p>{esc(reason)}</p><p>答案匹配：{esc(task.get("matched_text"))}；字符位置：{esc(task.get("answer_span"))}</p><details><summary>答案与重复片段/位置证据</summary><pre>{esc(_pretty(details))}</pre></details><details class="full-response"><summary>展开完整原始回答（非 600 字符摘录）</summary><pre>{esc(evidence.get("full_text", "未加载完整原文"))}</pre></details></td>')
            details = {'mechanism': row['mechanism'], 'calibration': row.get('calibration'),
                       'semantic_status': row['effect'].get('semantic_status'), 'pairing': row['pairing']}
            rows.append(f'<tr data-tags="{esc(tags)}"><th scope="row">{esc(row["probe_id"])}<small>{esc(cycle["cycle_id"])} · {esc(row["attribution"])}</small>{original_prompt}<p>{esc(", ".join(families))}</p><details><summary>本请求命中/校准</summary><pre>{esc(_pretty(details))}</pre></details></th>{"".join(cells)}</tr>')
    analysis = f'<details id="analysis" {"open" if mode == "analysis" else ""}><summary>Analysis Details · PPL / NLL / task / legacy classification</summary><pre>{esc(_pretty(data.get("analysis_details", {})))}</pre><p>仅作二级分析；PPL、RDR、VDR、L0–L6 不驱动 Distortion PASS。</p></details>'
    if data.get('numerical_effect'):
        analysis += '<details><summary>采样 NMSE / SQNR / 相对 L2（同一误差能量的派生，不改变判定）</summary><pre>' + esc(_pretty(data['numerical_effect'])) + '</pre></details>'
    filters = [("all", "全部"), ("distorted", "只看失常"), ("repetition", "重复"), ("structure", "结构"), ("garbage", "乱码"), ("unresolved", "答案未解"), ("crash", "错误/崩溃"), ("recovery_failure", "恢复失败")]
    options = "".join(f'<option value="{k}">{v}</option>' for k,v in filters)
    supplemental = '' if dataset else '<details><summary>补充任务的三项独立结论</summary><pre>' + esc(_pretty(data.get('normal_diagnostic_delivery', {}))) + '</pre></details>'
    display_note = '本次默认展示完整回答。' if dataset else '默认摘录最多 600 字符；可展开完整回答。'
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DemonCATPFI · Distortion Evaluation</title><style>
:root{{color-scheme:light dark;--bg:#f4f6fa;--fg:#182235;--card:#fff;--line:#d9e0eb;--accent:#175b83}}
@media(prefers-color-scheme:dark){{:root{{--bg:#101923;--fg:#e0e9f4;--card:#192634;--line:#334459;--accent:#8bc9e6}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{font-size:28px}}h2{{font-size:16px;letter-spacing:.03em}}.cards{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}article{{background:var(--card);padding:24px;border:1px solid var(--line);border-radius:10px}}dl{{display:grid;grid-template-columns:130px 1fr;gap:9px}}dt{{font-weight:600}}dd{{margin:0;overflow-wrap:anywhere}}.headline{{font-size:42px;font-weight:700;color:var(--accent)}}.headline span{{font-size:15px;display:block}}.badge{{display:inline-block;border:1px solid var(--line);padding:3px 7px;border-radius:4px;font-size:12px}}.matrix{{overflow:auto;margin-top:16px}}table{{border-collapse:collapse;width:100%;min-width:760px;table-layout:fixed}}th,td{{border:1px solid var(--line);vertical-align:top;text-align:left;padding:14px;background:var(--card)}}th:first-child{{width:17%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 ui-monospace,monospace}}small{{display:block;font-size:12px;font-weight:400;overflow-wrap:anywhere}}label{{display:block;margin:20px 0 8px}}select{{padding:8px;font:inherit}}details{{margin-top:12px}}summary{{cursor:pointer}}#count{{margin-left:16px}}[hidden]{{display:none!important}}@media(max-width:760px){{main{{padding:15px}}.cards{{grid-template-columns:1fr}}dl{{grid-template-columns:110px 1fr}}}}
article{{min-width:0;overflow-wrap:anywhere}}
</style></head><body><main><h1>DemonCATPFI · 故障可见效果</h1>
<p>{esc(d['probe_suite'])} · 固定检测器，无总分；UNKNOWN 不代表健康；未触发规则不代表语义正确。</p>
{supplemental}
{cards}{context_note}<label for="filter">按故障表现筛选</label><select id="filter">{options}</select><span id="count" aria-live="polite"></span>
<div class="matrix"><table><thead><tr><th>Probe / cycle</th><th>Baseline</th><th>Fault</th><th>Recovery</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<section><h2>恢复范围（必须同时阅读）</h2><pre>{esc(_pretty(d['recovery'].get('per_cycle_scopes', {})))}</pre></section>
<details><summary>稳定性与验收门禁</summary><pre>{esc(_pretty({'same_probe': d['same_probe_repeatability'], 'conditions': d['robust_conditions']}))}</pre></details>
{analysis if mode in ('analysis','both','distortion') else ''}
<p>{display_note}展示截短与 finish_reason=length 是两个独立事实。答案未交付不自动证明语义错乱；正确答案也不遮蔽后续循环。所有文本均经过 HTML 转义；本页不访问网络或控制模型。</p>
</main><script>const select=document.getElementById('filter');const rows=Array.from(document.querySelectorAll('tbody tr'));function filter(){{let n=0;for(const row of rows){{row.hidden=select.value!=='all'&&!row.dataset.tags.includes(select.value);if(!row.hidden)n++;}}document.getElementById('count').textContent=n+' / '+rows.length+' rows';}}select.addEventListener('change',filter);filter();</script></body></html>"""


def write_report(output, result, provenance, *, mode="distortion"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if 'single_request_effect' in result.to_dict():
        return _write_single_report(output, result, provenance)
    names = ["evaluation_result.json", "distortion_profile.json", "effect_card.json", "effect_card.md", "effect_dashboard.html", "distortion_hits.jsonl", "replay_manifest.json"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("result directory already contains report files; never overwrite historical evidence")
    data = result.to_dict()
    d = data["distortion"]
    card = {"fault_configuration": data["fault_configuration"], "mechanism_gate": d["mechanism_gate"],
            "normal_diagnostic_delivery": data.get('normal_diagnostic_delivery'),
            "visible_effect": {key: d[key] for key in ("probe_suite", "probe_count", "complete_cycles", "stable_distorted_probes", "stable_phenotype_families", "same_probe_repeatability", "crash_count", "recovery", "robust_gate", "robust_conditions")}}
    if data.get('dataset_delivery'):
        card['dataset_delivery'] = data['dataset_delivery']
    if data.get('model_targeting'):
        card['model_targeting'] = data['model_targeting']
    for name, value in ((names[0], data), (names[1], d), (names[2], card), (names[6], provenance)):
        (output / name).write_text(_pretty(value)+"\n", encoding="utf-8")
    config = data["fault_configuration"]
    md = f"""# 故障可见效果卡

## 注入了多少、在哪里注入

- Capability：`{config['capability']}`；策略：`{config['strategy']}`。
- 模块族：`{', '.join(config['module_families'])}`；层：{len(config['targeted_layers'])}/{config['available_layers']}；rank：{config['targeted_ranks']}/TP{config['world_size']}。
- Strength：`{config['strength']}`。
- Hit policy：`{config['hit_policy']}`。
- Mechanism：**{d['mechanism_gate']}**，{d['mechanism_valid_requests']}/{d['mechanism_requests']} 个原始请求通过。

## 用户实际看到了什么

- 数据集：`{d['probe_suite']}`；版本/hash 见 manifest。
- 完整周期：{d['complete_cycles']}。
- 稳定失常：**{len(d['stable_distorted_probes'])}/{d['probe_count']}**；样本：`{d['stable_distorted_probes']}`。
- 稳定 phenotype：`{ {k: len(v) for k,v in d['stable_phenotype_families'].items()} }`。
- Crash：{d['crash_count']}。
- 失常表型恢复：{d['recovery']['restored_pairs']}/{d['recovery']['fault_induced_pairs']}；该分母不是全测试集。
- 全量恢复 Gate：{d['recovery']['gate']}，同时核对全用例任务/validator 状态及 runtime。
- **{d['robust_gate']}**；未通过条件：`{[k for k,v in d['robust_conditions'].items() if not v]}`。

## 补充任务的独立结论

`{data.get('normal_diagnostic_delivery', {})}`

旧 robust breadth gate 仍保留；本轮可交付用例不要求产生两种病态。未命中、无数值变化和证据未知保留在完整矩阵，不伪装为“无效果”。

## 各周期恢复范围

```json
{_pretty(d['recovery'].get('per_cycle_scopes', {}))}
```

Scorer：`{d.get('scorer_version')}`。任务结果、生成终止、输出病态和机制归因分别记录；仅 unresolved 不计为明显病态。

输出轨迹变化不等于病态退化。N 个周期必须是同一道题稳定失常；不将不同题的数量相同视为可重复。
本结果是固定 smoke 集与规则证据，不是统计显著性或总体模型性能结论。只检测字符级乱码；不推断语义不连贯。

## 文件

- [逐项矩阵](effect_dashboard.html)
- [完整评测结果](evaluation_result.json)
- [只包含有效失常 pair 的日志](distortion_hits.jsonl)
- [原始来源与校验信息](replay_manifest.json)

PPL/NLL、旧 RDR/VDR/L0–L6 保留在 Dashboard 的 Analysis Details；不决定本页 PASS。
"""
    if data.get('dataset_delivery'):
        delivery = data['dataset_delivery']
        target, policy = _dataset_configuration(config)
        md = f'''# MT-Bench 官方首轮定向子集：FakeRequant 有损集成

## 两个独立结论

- DATASET_BACKED_LOSSY_INTEGRATION：{delivery['DATASET_BACKED_LOSSY_INTEGRATION']}
- VISIBLE_DISTORTION_OBSERVATION：{delivery['VISIBLE_DISTORTION_OBSERVATION']}
- 数据：lm-sys/FastChat @ {delivery['dataset']['revision']}；问题 ID：{delivery['question_ids']}，turn 0。
- 机制：{d['mechanism_gate']}；hit={d['measurement_hits']}；changed elements={d['changed_elements']}。
- 真实目标：{target}
- {policy}
- 输出分叉：{delivery['response_divergence']}/{d['probe_count']}；可见异常样本：{delivery['visible_sample_ids']}。
- 全文精确恢复：{delivery['exact_recovery']}/{d['probe_count']}；runtime recovery：{d['recovery']['gate']}。
- Crash：{d['crash_count']}；完整周期 N=1，不声称稳定性或统计显著性。
- 旧 16-probe / 3-cycle robust gate：NOT_APPLICABLE，不以它判断本轮有损集成失败。

任务结果、终止原因、数值变化和输出病态分别报告。没有开放式任务 validator 或 judge，
不计算官方 MT-Bench 得分；所有样本 SEMANTIC_REVIEW_REQUIRED。未检测到规则病态不等于语义正确。

## 完整输出与证据

- [逐样本完整 Baseline / Fault / Recovery 并排](effect_dashboard.html)
- [完整 EvaluationResult、重复片段及位置](evaluation_result.json)
- [数据来源与复现信息](replay_manifest.json)

Scorer：{d['scorer_version']}，hash 见 evaluation_result.json；阈值在任何 fault 前冻结。
原始请求/响应保持完整，Dashboard 本轮默认显示全文，不用 600 字符摘录替代。
'''
        if data.get('model_targeting'):
            targeting = data['model_targeting']
            window = targeting.get('window', {})
            md += ('\n## 自动目标计划验收\n\n'
                   f"- plan_sha256：`{targeting.get('plan_sha256')}`。\n"
                   f"- 总状态：{targeting.get('status', data.get('status'))}。\n"
                   f"- Mechanism：{window.get('mechanism')}。\n"
                   f"- Target call window：{window.get('target_call_window')}。\n"
                   f"- Full request window：{window.get('full_request_window')}。\n"
                   f"- 精确恢复：{targeting.get('recovery_exact')}。\n")
        if data.get('rule_context_review'):
            md += '\n## 告警上下文（不修改原分数）\n\n```json\n' + _pretty(data['rule_context_review']) + '\n```\n'
    (output / names[3]).write_text(md, encoding="utf-8")
    (output / names[4]).write_text(render_dashboard(result, mode=mode), encoding="utf-8")
    with (output / names[5]).open("w", encoding="utf-8") as handle:
        for cycle in d["cycles"]:
            for row in cycle["matrix"]:
                if row["visible_distortion"]:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    return {name: str(output / name) for name in names}


def _single_dashboard(data):
    """Same result renderer, a single-call lifecycle view; never a control UI."""
    esc = lambda v: html.escape(str(v), quote=True).replace('\r', '&#13;').replace('\n', '&#10;')
    single = data['single_request_effect']
    config, effect = data['fault_configuration'], single['effect']
    numeric = data['numerical_effect'].get('aggregate', {})
    summary = {'目标 rank': config.get('targeted_ranks'), '每 worker 模块数': len(config.get('actual_module_names', [])),
        '强度': config.get('strength'), '机制': single['window']['mechanism'],
        '目标调用覆盖': single['window']['target_call_window'], '完整请求覆盖': single['window']['full_request_window'],
        '归因有效': single['eligible'], '全文分叉': effect['response_diverged'],
        '首分叉': effect['first_divergence'], '比较单位': effect['comparison_basis'],
        '规则结论': effect['display_status'], '采样 NMSE': numeric.get('nmse'),
        '采样 SQNR dB': numeric.get('sqnr_db'), '采样状态': numeric.get('status')}
    columns = ''.join('<article><h2>' + title + '</h2><pre>' +
        esc(single['raw'][phase].get('response_text')) + '</pre><details><summary>原始记录 / token pieces</summary><pre>' +
        esc(_pretty(single['raw'][phase])) + '</pre></details></article>'
        for phase, title in [('baseline', '正常对照'), ('fault', '有损输出')])
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>单次 FakeRequant 效果</title>' + \
        '<style>body{font:16px system-ui;max-width:1280px;margin:auto;padding:24px;background:#f5f7fa;color:#182b40}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}article{background:white;padding:20px;border:1px solid #ccd5df}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px monospace}summary{cursor:pointer}h1{font-size:26px}</style>' + \
        '<h1>单次有损推理 · 结果只读</h1><p>不是完整 B/F/R 周期。停止实例记录见上一级产物索引；热卸载恢复与重启后文本恢复均未测。</p>' + \
        '<h2>计划与实际效果</h2><pre>' + esc(_pretty(summary)) + '</pre>' + \
        '<section>' + columns + '</section><details><summary>完整配置、各 worker 覆盖、规则片段与数值</summary><pre>' + \
        esc(_pretty({'config': config, 'window': single['window'], 'effect': effect,
                     'numerical_effect': data['numerical_effect']})) + '</pre></details>' + \
        '<h2>生命周期</h2><pre>' + esc(_pretty(data['instance_lifecycle'])) + \
        '</pre><p>轨迹分叉不等于严重病态；规则告警不等于开放式语义正确性认证。无 recovery 文本，不填充为 baseline。</p></html>'


def _write_single_report(output, result, provenance):
    data = result.to_dict()
    names = ['evaluation_result.json', 'effect_profile.json', 'effect_card.md', 'effect_dashboard.html', 'replay_manifest.json']
    if any((output / name).exists() for name in names):
        raise FileExistsError('Never overwrite historical report')
    for name, value in [('evaluation_result.json', data),
                        ('effect_profile.json', {k: data[k] for k in ('mechanism', 'single_request_effect', 'numerical_effect', 'instance_lifecycle')}),
                        ('replay_manifest.json', provenance)]:
        (output / name).write_text(_pretty(value) + '\n', encoding='utf-8')
    (output / 'effect_dashboard.html').write_text(render_dashboard(result), encoding='utf-8')
    (output / 'effect_card.md').write_text('# 单次 FakeRequant 效果\n\n'
        '本结果保留完整正常/有损输出，不伪造 Recovery。完整 worker 停止确认见宿主产物索引。\n\n'
        '- [完整原文与指标](effect_dashboard.html)\n- [EvaluationResult](evaluation_result.json)\n\n'
        '热卸载恢复：NOT_MEASURED；重启后文本对照：NOT_MEASURED；完整周期 N=0。\n', encoding='utf-8')
    return {name: str(output / name) for name in names}
