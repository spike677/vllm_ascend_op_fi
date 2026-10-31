# 全模型策略注入：实现与验收边界

2026-10-31，正式 `formal_stream_a1` 的 B1/B8 五轮完整模型验收已通过，联合结论为 `PASS_MODEL_ALL_ELIGIBLE_NATIVE_STRATEGY_COVERAGE`，报告 SHA256 为 `16e82c3e072442839465a4e2b604d891430ead77bc2aa018a8f7ce9fe6fed8cf`。本文说明已验收的实现与证据边界；当前仅准备发布，不再重跑 NPU。B1/B8分别在各自模型进程完成五轮。

顺序 reader 新版本已完成 `pilot_stream_a1` 的56次原始 SDPA 校准准入，联合校准报告 SHA256 为 `37e73c33cf47b3023404d17d471ae3831d0219ecde0be6291a7efeaa2994e465`。该校准未开启故障，不计入随后已独立通过的 `formal_stream_a1` 正式560次调用。

## 模型入口与原生调用链

[model_probe.py](../model_probe.py) 的 `main()` 使用原生 Transformers 加载未量化 Qwen3-0.6B checkpoint，明确指定 FP16、`attn_implementation='sdpa'` 和 `eval()`。完整 28 层、输出投影、残差和 `lm_head` 均正常执行。模型与分词器文件、安装版本及源码、私有 vendor、通用 Python 核心和证书均固定并核验 SHA。

```text
冻结的 input_ids / position_ids，真实 KV cache
  → 原始 Qwen3Attention.forward
  → F.scaled_dot_product_attention 的共同观察包装器
  → ModelObserver.observe
  → StrategySession.execute（native / controlled / controlled / off）
  → FlashAttentionAdapter.native_scope
  → invoke_original → 原始 F.scaled_dot_product_attention（恰好一次）
  → NPU SDPA backend → npu_fusion_attention_v3 → aclnnFlashAttentionScore
  → 原生 host tiling 与 kernel → 原始返回对象 → 模型后续计算
```

调用链中的 native FA 路径必须由真实 launch 证明；Python SDPA 名称或 `FLASH_ATTENTION` 上下文均不能单独证明实际 kernel。这里没有替换 attention 算法、返回值或 Q/K/V，也没有用替换 Python `torch_npu.npu_fusion_attention_v3` 名称来假定截获了内部 C++ 调用。

## 每层基线与每次匹配

`ModelObserver.__init__()` 为每个实际 `self_attn` 创建独立 adapter、Session 和 disabled-policy Session。几何相同的其他层不能覆盖本层基线。正式模式依次为 `calibration → native → controlled → controlled → off`；两次 `controlled` 由 `run_index` 区分。

校准在 `ModelObserver.observe()` 中走独立的 native-only 观察作用域。[model_context.py](../model_context.py) 的 `probe_template()` 明确标记 **UNVERIFIED intent**，只负责让原生观察器进入作用域，不证明调用符合条件。原调用返回后，`actual_native_call()` 从原始 host 公共 IR19/attrs、同调用的 launch raw 和 205 项 POD 重建实际上下文；再由 `prefill_context()` 核对当前 Python Q/K/V、GQA、causal、dropout 和 scale。只有这些检查及 adapter 解析通过，才缓存真实基线。

后续 native、两次 controlled 和 off 共用同一个 `StrategySession.execute()` 调用位置和原始 SDPA 调用位置。`Policy.evaluate()` 每次评估实际上下文和合法候选，没有首次命中上限。本轮只使用证书中的 `joint` 选择：预期 native M128/R8 与候选 M64/R4；标签或 decision 只表示意图，实际是否切换由真实完整执行计划比较决定。controlled 观察不会覆盖 native 基线；off 使用同层 adapter 和关闭的 policy。

关键实现位置：

| 文件 / 函数 | 职责 |
| --- | --- |
| [session.py](../../../native_execution_strategy/session.py) / `StrategySession.execute` | 共同执行路径；原调用一次；保留返回对象；观察失败不伪造切换或数值结论。 |
| [policies.py](../../../native_execution_strategy/policies.py) / `Policy.evaluate` | `every_match` 与合法替代策略选择。 |
| [flash_attention.py](../../../native_execution_strategy/adapters/flash_attention.py) / `observe_current`、`native_scope`、`parse_native_evidence` | 本上下文基线、PID/TID/token 绑定作用域、实际原生证据及证书核验。 |
| [fa_plan.h](../../../native_execution_strategy/native/fa_plan.h) / `DecideFactors`；[fa_bridge.h](../../../native_execution_strategy/native/fa_bridge.h) / `SelectFactors` | 核对实际原生上下文及 native factors 后选择 M/R。 |
| [apply_fa_bridge.py](../../../native_execution_strategy/native/apply_fa_bridge.py) / `main` | 在匹配版本 host 的原生 factor 选择位置接入 bridge，后续由原生 tiling 计算派生计划；不直接覆盖已序列化 tiling buffer，不修改 kernel。 |

本轮沿用 `cann9.1-s1s2-v1` validated 证书与对应构建。候选支持范围是 B1/B8、FP16 BNSD、Q16/KV8 heads、D128、S1=S2=2048 及固定 causal/precision 等条件。完整计划含 166 项 typed 执行字段和三个原始 64-byte section，共 169 项；另核 34 项语义输入契约，不只核 M/R 或 kernel key。

## 请求、cache 与 decode

B1 使用一个目标请求，B8 使用同一目标加七个不同的冻结请求。每个请求以 `request_id`、`batch_slot`、token SHA、长度及 query positions 映射到实际 batch row。`request_rows()` 与每阶段 `request_mapping.json` 记录映射，实际 `input_ids` 和 `position_ids` 在模型调用前后均核验。

每轮从 `cache=None` 开始，执行 2048-token prefill，再用该轮真实产生的 KV cache 和冻结的后续 token 执行一次 decode：Q1、KV2049、位置2048。不同轮不复用 cache，也不把各轮生成的 argmax 作为不同 decode 输入。因此这是固定前缀下的模型传播检验，不是自由生成质量评估。

**decode 超出证书形状范围。** 当前 `ModelObserver.observe()` 显式检查没有遗留 NES 环境作用域，然后直接调用 `invoke_original()`，记录 unsupported、原始返回和真实执行路线。这是原始调用透传，**没有经过 `StrategySession` 的 unsupported 分支**，不能据此宣称该分支也完成了实机验收。无 native attention dispatch 的日志窗口也不能自行命名为 math 路径。

## 覆盖分母与实际策略证明

分母先由 [audit_model_coverage.py](audit_model_coverage.py) 的 `audit()` 从原始 msprobe SDPA 节点、construct 父模块、源码栈、实际模型配置及无条件 module pre/post hooks 建立。再由 [native_launch_census.py](native_launch_census.py) 的 `census()` 扫描整段原始日志，核查所有 attention dispatch 与调用窗口，不能只读取命中日志。

[accept_model_execution.py](accept_model_execution.py) 按 step、layer、module call ID、SDPA call ID 和请求映射 SHA 将两套分母与每次原始记录一一绑定。对所有 prefill candidate 核真实公共 context、原 host 事件及完整 launch raw。它还从整份原日志的原始 byte/line 范围重新解析 tiling，核窗口与前缀 SHA；host 与派生 JSON 互相一致但不符合实际日志的情况不能通过。缺 host、guard 或实际 raw 是 **UNKNOWN 阻断项**，不能归为“不满足条件”后排除。

正式实采分母与预注册逐项一致：560次 SDPA，包括280次 prefill和280次 unsupported decode；两个 controlled run 的符合条件 prefill 已112/112次实际切换，其余168次 prefill 保持 native。受控 decode 共112次，均原 API 透传。受控 layer/request 映射已核验504对，覆盖全部28层和8个不同请求。B8一次调用承载8个请求，不能把映射对说成504次独立 kernel 命中。上述数字来自实际分母与完整执行计划核验，不以计划或命中标签代替。

### 原始日志的顺序读取

采集端使用 [model_native_stream.py](../model_native_stream.py) 的单实例 `NativeLogCursor`，由 `ModelObserver` 持有，通过 [native_evidence.py](../native_evidence.py) 的 `verified_record(..., launch_reader=...)` 接入。每次只读取尚未消费的日志，直到对应 prefill 的原始 exit marker，保存该调用窗口的原字节；raw 仍交给未修改的 `actual_launch()` 进行精确字符拼接和解析。没有从期望 M/R 构造 raw，也没有改变原始调用或统计数据。

cursor 保持文件 identity、全局 byte/line offset、滚动 prefix SHA 和原始 library-load 记录，拒绝已观察到的 inode 替换、截断、重复或倒序调用、跨调用及 PID/TID/token 错配。两个 prefill 之间的 decode 日志也在下次读取时纳入 prefix；失败时保留已写原窗口，禁止继续使用失效 cursor。`result.native_log_reader` 记录实际 reader 模式、源码 SHA、消费字节和完成窗口数量，但不充当独立通过证据。

该优化不反复重哈希已经消费的字节，**不能单独检测旧前缀的同长度改写**。最后一个 prefill 之后的 decode 也不由后续 prefill 读取。因此，最终独立验收仍从整份原日志重算 prefix、窗口范围和 raw，整段 native census 仍须覆盖最后一次 decode；不能用 cursor 回执替代这两步。新 reader 已在旧实采日志的66个窗口上完成 CPU 逐字段等价检查；这证明解析一致性，不是正式故障覆盖结论。运行预算以当次冻结 case/controller 配置及资源回执为准。

## 数值、统计首差与恢复

采集配置为 `task=statistics / level=mix / summary_mode=statistics`，使用明确披露的 task-local FP32 high collector。原始 dump/stack/construct 不改写；[model_helpers.py](../model_helpers.py) 的 `tensor_snapshot()` 诊断读回仍保留在原始 trace 中，只按该精确源码栈分类。统计目录不含 `.pt` 或 tensor bin；另外的完整 logits 二进制仅用于私有逐字节核验。

[compare_model_statistics.py](compare_model_statistics.py) 的 `compare_phase()` 只从原始统计及调用栈推断首个被观测数值差异，不读取 host 命中、计划或私有张量来指定位置。calibration 的手动 scope 与 Session 栈可能不同，不通过删帧强行对齐。null 仍是不可观测；四项摘要相同不证明逐元素相等；SDPA 边界差异不等同于定位到某个 CANN 内部归约指令。

联合验收另行核对每个原调用输入 before/after 的完整逻辑 SHA、身份与几何，仅允许已明确记录且数值未变的 format 0→2 规范化。按请求槽比较各层输出 SHA，并对最终 logits 读取完整私有字节：calibration/native 和 native/off 应恢复一致；两次 controlled 应稳定。后层 B/F 输入允许因前层改变而不同，不能把每层输出差都解释为该层独立数值起点。

策略实际切换、数值可观测变化和模型 logits 变化分别报告。全部 eligible 调用命中不要求每层或每个请求都产生可见差异；logits 不同也不能直接推出准确率恶化、历史缺陷复现或 vLLM FIA 服务已完成接入。

正式统计结果中，B1/B8的两次fault共四组独立首差均为 prefill第0层 SDPA output Mean。两次controlled重复稳定，off后各层及逐请求输出与完整logits恢复。该结论限于已验收矩阵，不扩大证书形状范围或把下游传播解释为每层独立数值起因。
