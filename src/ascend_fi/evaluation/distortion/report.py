"""Standalone escaped HTML matrix and machine-readable effect artifacts."""

from __future__ import annotations

import html
import json
from pathlib import Path


def _pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def render_dashboard(result, *, mode="distortion"):
    data = result.to_dict()
    d, config = data["distortion"], data["fault_configuration"]
    esc = lambda v: html.escape(str(v), quote=True)
    n, p = d["total_cycles"], d["probe_count"]
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
<dt>Crash</dt><dd>{d['crash_count']}</dd><dt>Recovery</dt><dd>{d['recovery']['restored_pairs']} / {d['recovery']['fault_induced_pairs']} pairs · {d['recovery']['gate']}</dd></dl>
<p class="badge">{esc(d['robust_gate'])}</p></article></section>"""
    rows = []
    for cycle in d["cycles"]:
        for row in cycle["matrix"]:
            families = row["effect"]["families"]
            tags = ("distorted " if row["visible_distortion"] else "") + " ".join(families).lower()
            if row["effect"]["verdict"] == "CRASH":
                tags += " crash"
            if row["recovery_restored"] is False:
                tags += " recovery_failure"
            cells = []
            for phase in ("baseline", "fault", "recovery"):
                evidence = row[phase]
                verdict = row["effect"]["verdict"] if phase == "fault" else row["recovery_effect"]["verdict"] if phase == "recovery" else "BASELINE_FAILURE" if row["effect"]["baseline_failed_detectors"] else "REFERENCE"
                text = evidence["text"] if evidence["text"] is not None else "UNKNOWN: raw output missing"
                reason = "; ".join(x["reason"] for x in row["effect"]["detectors"] if x["triggered"] is True) if phase == "fault" else ""
                cells.append(f'<td><span class="badge">{esc(verdict or "UNKNOWN")}</span><pre>{esc(text)}</pre><p>{esc(reason)}</p><details><summary>证据</summary><pre>{esc(_pretty(evidence))}</pre></details></td>')
            rows.append(f'<tr data-tags="{esc(tags)}"><th scope="row">{esc(row["probe_id"])}<small>{esc(cycle["cycle_id"])} · {esc(row["attribution"])}</small><p>{esc(", ".join(families))}</p></th>{"".join(cells)}</tr>')
    analysis = f'<details id="analysis" {"open" if mode == "analysis" else ""}><summary>Analysis Details · PPL / NLL / task / legacy classification</summary><pre>{esc(_pretty(data.get("analysis_details", {})))}</pre><p>仅作二级分析；PPL、RDR、VDR、L0–L6 不驱动 Distortion PASS。</p></details>'
    filters = [("all", "全部"), ("distorted", "只看失常"), ("repetition", "重复"), ("structure", "结构"), ("garbage", "乱码"), ("unresolved", "答案未解"), ("crash", "错误/崩溃"), ("recovery_failure", "恢复失败")]
    options = "".join(f'<option value="{k}">{v}</option>' for k,v in filters)
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DemonCATPFI · Distortion Evaluation</title><style>
:root{{color-scheme:light dark;--bg:#f4f6fa;--fg:#182235;--card:#fff;--line:#d9e0eb;--accent:#175b83}}
@media(prefers-color-scheme:dark){{:root{{--bg:#101923;--fg:#e0e9f4;--card:#192634;--line:#334459;--accent:#8bc9e6}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{font-size:28px}}h2{{font-size:16px;letter-spacing:.03em}}.cards{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}article{{background:var(--card);padding:24px;border:1px solid var(--line);border-radius:10px}}dl{{display:grid;grid-template-columns:130px 1fr;gap:9px}}dt{{font-weight:600}}dd{{margin:0;overflow-wrap:anywhere}}.headline{{font-size:42px;font-weight:700;color:var(--accent)}}.headline span{{font-size:15px;display:block}}.badge{{display:inline-block;border:1px solid var(--line);padding:3px 7px;border-radius:4px;font-size:12px}}.matrix{{overflow:auto;margin-top:16px}}table{{border-collapse:collapse;width:100%;min-width:760px;table-layout:fixed}}th,td{{border:1px solid var(--line);vertical-align:top;text-align:left;padding:14px;background:var(--card)}}th:first-child{{width:17%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 ui-monospace,monospace}}small{{display:block;font-size:12px;font-weight:400;overflow-wrap:anywhere}}label{{display:block;margin:20px 0 8px}}select{{padding:8px;font:inherit}}details{{margin-top:12px}}summary{{cursor:pointer}}#count{{margin-left:16px}}[hidden]{{display:none!important}}@media(max-width:760px){{main{{padding:15px}}.cards{{grid-template-columns:1fr}}dl{{grid-template-columns:110px 1fr}}}}
</style></head><body><main><h1>DemonCATPFI · 故障可见效果</h1>
<p>{esc(d['probe_suite'])} · 固定检测器，无总分；UNKNOWN 不代表健康。历史集合不是新 16-probe suite。</p>
{cards}<label for="filter">按故障表现筛选</label><select id="filter">{options}</select><span id="count" aria-live="polite"></span>
<div class="matrix"><table><thead><tr><th>Probe / cycle</th><th>Baseline</th><th>Fault</th><th>Recovery</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<details><summary>稳定性与验收门禁</summary><pre>{esc(_pretty({'same_probe': d['same_probe_repeatability'], 'conditions': d['robust_conditions']}))}</pre></details>
{analysis if mode in ('analysis','both','distortion') else ''}
<p>文字摘录最多 600 字符；完整记录见单元格 source_ref。输出中的文本均经过 HTML 转义；本页不访问网络或控制模型。</p>
</main><script>const select=document.getElementById('filter');const rows=Array.from(document.querySelectorAll('tbody tr'));function filter(){{let n=0;for(const row of rows){{row.hidden=select.value!=='all'&&!row.dataset.tags.includes(select.value);if(!row.hidden)n++;}}document.getElementById('count').textContent=n+' / '+rows.length+' rows';}}select.addEventListener('change',filter);filter();</script></body></html>"""


def write_report(output, result, provenance, *, mode="distortion"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ["evaluation_result.json", "distortion_profile.json", "effect_card.json", "effect_card.md", "effect_dashboard.html", "distortion_hits.jsonl", "replay_manifest.json"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("result directory already contains report files; never overwrite historical evidence")
    data = result.to_dict()
    d = data["distortion"]
    card = {"fault_configuration": data["fault_configuration"], "mechanism_gate": d["mechanism_gate"],
            "visible_effect": {key: d[key] for key in ("probe_suite", "probe_count", "complete_cycles", "stable_distorted_probes", "stable_phenotype_families", "same_probe_repeatability", "crash_count", "recovery", "robust_gate", "robust_conditions")}}
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

- 数据集：`{d['probe_suite']}`，不是伪装的新 16 条探针。
- 完整周期：{d['complete_cycles']}。
- 稳定失常：**{len(d['stable_distorted_probes'])}/{d['probe_count']}**；样本：`{d['stable_distorted_probes']}`。
- 稳定 phenotype：`{ {k: len(v) for k,v in d['stable_phenotype_families'].items()} }`。
- Crash：{d['crash_count']}。
- Recovery：{d['recovery']['restored_pairs']}/{d['recovery']['fault_induced_pairs']} 个故障引起的失常 pair 恢复；gate={d['recovery']['gate']}。
- **{d['robust_gate']}**；未通过条件：`{[k for k,v in d['robust_conditions'].items() if not v]}`。

输出轨迹变化不等于病态退化。N 个周期必须是同一道题稳定失常；不将不同题的数量相同视为可重复。
本结果是固定 smoke 集与规则证据，不是统计显著性或总体模型性能结论。只检测字符级乱码；不推断语义不连贯。

## 文件

- [逐项矩阵](effect_dashboard.html)
- [完整评测结果](evaluation_result.json)
- [只包含有效失常 pair 的日志](distortion_hits.jsonl)
- [原始来源与校验信息](replay_manifest.json)

PPL/NLL、旧 RDR/VDR/L0–L6 保留在 Dashboard 的 Analysis Details；不决定本页 PASS。
"""
    (output / names[3]).write_text(md, encoding="utf-8")
    (output / names[4]).write_text(render_dashboard(result, mode=mode), encoding="utf-8")
    with (output / names[5]).open("w", encoding="utf-8") as handle:
        for cycle in d["cycles"]:
            for row in cycle["matrix"]:
                if row["visible_distortion"]:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    return {name: str(output / name) for name in names}
