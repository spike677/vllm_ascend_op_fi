# 完整模型独立覆盖与统计分析合同

本目录只实现只读分析；不修改模型、通用策略核心、证书、collector 或旧实验，不调度设备。
2026-10-31，`formal_stream_a1` 的完整 B1/B8 实测联合验收已通过，状态为 `PASS_MODEL_ALL_ELIGIBLE_NATIVE_STRATEGY_COVERAGE`。联合报告 SHA256：
`16e82c3e072442839465a4e2b604d891430ead77bc2aa018a8f7ce9fe6fed8cf`。
实际560次SDPA、112/112次受控prefill切换、112次受控decode透传及504个layer/request映射均已核验；四组fault首差均为第0层SDPA输出Mean，重复和off恢复通过。当前仅准备发布，不再重跑NPU。
下文保留预注册规则和分析接口；最新实测结论及范围见 [实验README](../README.md)，不以CPU fixture替代实机证据。

## 固定矩阵

两个 case：B1/P0 与 B8/P0…P7；八个 prompt 的冻结 token 序列不同且各 2048 token。
每 case 五次真实模型运行：`calibration`、`native`、`fault_a`、`fault_b`、`off`。
`fault_a/b` 是分析标签，harness 两次运行的实际 mode 都是 `controlled`，由 run_index 2/3 区分。
每次运行先 prefill，再用本次产生的 KV cache 和冻结 teacher-forced token 执行一次 decode。
禁止用上次运行的 cache，或把模型 argmax 当作变化的 decode 输入。

建议 `msprobe_step = 2 * run_index + phase_index`，prefill 为 0，decode 为 1。
每 case 共 10 个统计 step，每 step 按模型 config 和实际模块清单核验 28 个 SDPA 调用。
计划总计 560 次 SDPA；其中两个 fault run 的 prefill 共 112 次候选受控调用，decode 共
112 次必须单列的 unsupported/passthrough 调用。候选 prefill 是否均落在实际原生 FA
路径，需要真实 launch 证据确认，不能仅由 Python API 名称推断。

B8 每次原生调用承载八个 batch row，仍只计一次调用。每个 fault prefill 的 layer/request
映射为 B1 的 28 与 B8 的 224；两次 fault 合计 504 个映射对，不称为 504 次 kernel 命中。

## 最小采集接口提案

1. 冻结 case/model 输入：`case_id`、`model_config` 及 SHA、28 层模块清单、请求顺序、各请求
   唯一 `request_id`/`batch_slot`、prefill token SHA、固定 decode token，及完整冻结输入文件。
2. `phase_manifest.json`：10 行，每行包含 `run_index`、`run_id`、`mode`、`phase`、
   `msprobe_step`、实际 request 顺序、实际 input_ids SHA、query/past/total token 数、
   `cache_source_run_id`。该文件描述真实模型输入，不含策略命中字段。
3. `module_events.jsonl`：无条件安装于全部 `self_attn` 模块的独立 pre/post hook。
   每条含 `event=pre|post`、单调 `event_index`、`call_id`、run/phase/step、真实
   `module_path`/`layer_index`、hidden/output shape、dtype、request 顺序。
   pre/post 共用 call_id；不能按策略 eligible/命中结果过滤事件。
4. 原始 `config.json`、`receipt.json`、`collector_provenance.json` 和
   `observation/step0…9/rank0/{dump,stack,construct}.json`；保持
   task=statistics、level=mix、summary_mode=statistics，无完整张量文件。
   延续已披露 task-local FP32 high collector，实际 import 路径和源码 SHA 必须绑定。
5. 外部验收证据：每次 SDPA 的实际原生 launch/POD、Session observation/decision 和
   unsupported 原因。它们只用于核对实际执行，不用于建立分母或决定统计首差。

`result.json` 必须将 phase manifest、模块清单、模块事件及 receipt 的 SHA 封存；
controller completion 再绑定 result。模型 hidden_size 从实际 config 读取：当前 checkpoint
为 1024，不能将 Q projection 的 16×128=2048 误当作 self_attn hidden/output 宽度。

```bash
python3 -B analysis/audit_model_coverage.py \
  --case-dir outputs/ATTEMPT/CASE --completion outputs/ATTEMPT/completion.json \
  --model-config /home/gyl/models/Qwen3-0.6B/config.json \
  --output outputs/analysis/CASE_framework_coverage.json
python3 -B analysis/compare_model_statistics.py \
  --case-dir outputs/ATTEMPT/CASE --completion outputs/ATTEMPT/completion.json \
  --output outputs/analysis/CASE_statistics
```

两工具仅读原始证据并写全新报告。Pilot 两 step 只做 schema/覆盖检查，不产生正式对照结论。

## 独立覆盖分母

按原始 msprobe API 枚举 `Functional.scaled_dot_product_attention.*.forward`，沿 construct
找到真实 `Module.model.layers.i.self_attn.Qwen3Attention...`；用 model config、模块清单和
独立 pre/post hooks 交叉核对。键为 `(case, run, phase, layer, call_ordinal)`。
缺失、重复、无 post、无父节点、请求顺序或形状不一致均报告失败，不用 fault 日志补齐。

从已核对的实际 BNSD shape 与冻结输入映射展开 layer/request 对，单列
`batch_row_coverage`。整批 Max/Min/Mean/Norm 不能分解为每请求摘要，故
`per_request_tensor_difference` 默认不可观测。模型末 token 的逐请求 logits 可作另一份
模型影响证据，不能冒充原生 msprobe 的逐请求 FA 统计。

分母先完整记录全部 SDPA 调用，再将真实上下文和支持合同用于分类：prefill 候选、decode
unsupported、其他未确认路径。之后才与 native launch 和 Session 结果一一核对，统计
真实策略切换分子。SDPA 总数、原生 FA 已证数量和受控命中数量为不同字段。

## 统计比较与结论边界

每 case 固定五组完整运行比较：calibration→native、native→fault_a、native→fault_b、
fault_a→fault_b、native→off；两 phase 分别对齐后按 prefill→decode 顺序给出一次全模型
首差。即 10 个完整运行比较、20 个 phase 比较；不按结果挑选层或请求。

首差仅读 config/dump/stack/construct；receipt/completion/collector 只作完整性门槛。
保留完整模型调用顺序、层级和 source stack。若只读诊断节点被捕获，保留原记录，按经
源码审核的明确函数列表标注，不能把真实 SDPA 因含 observer frame 而丢弃。

全层同时注入后，早期输出差会传播为后续层输入差。逐层报告 `input_already_different`、
`output_difference`、`unobservable`，不得要求所有后续层跨 arm 输入相同，也不得把每层
都称为独立故障起点。每次调用内部 before/after 输入不变属于独立完整性检查。

null、统计碰撞与辅助返回变化须明确保留。层/请求覆盖、实际策略切换、数值差异和模型
影响分别验收；未出现可观测差异也是有效结果，不制造成功结论。
