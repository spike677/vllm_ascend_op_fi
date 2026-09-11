# 算法与状态逻辑：以固定源码为准

本页解释当前实现，不是新算法设计。Python工具基于Qwen已测试提交；参考依赖固定到[源码及SHA](../../experiments/fake_requant_reference/README.md)。指标只在[字段字典](../evaluation/fake_requant_tool_v1_metrics_zh.md)定义，产物见[合同](../reference/artifact_io_contracts_zh.md)。

## 1. 模型识别与角色解析

[model_descriptor.py](../../src/ascend_fi/inspect/model_descriptor.py)读取真实模型类、config（含嵌套text config）、模块枚举、rank来源和执行参数，输出service epoch、model/runtime fingerprint和枚举完整性。环境缺失字段保留UNKNOWN，不用镜像tag猜包版本。

[model_adapters.py](../../src/ascend_fi/inspect/model_adapters.py)结合模型类/config、实际模块/父模块类、层号和路径解析角色；Qwen2/3 dense与MoE/DeepSeek有不同资格边界。只读规则命中不是已认证。枚举截断、未知结构、缺角色保留blockers；按模型层数核对预期角色集合，不以“找到多少”自行缩小分母。

Qwen0.6B实际28层，每层ATTN_QKV、ATTN_OUT、MLP_GATE_UP、MLP_DOWN，共112。融合入口只挂一次；不同时选择父/子或假想拆分q/k/v导致重复。角色是模型语义，底层kernel对应关系另需证据。

## 2. 真实实参绑定与计划

[session.py](../../src/ascend_fi/inspect/session.py)安装返回None的元数据pre-hook，不拷贝activation本体；只读不等于没有同步/调度影响。默认观察60秒、256 events、1 MiB、每模块2样本；上限300秒、2048 events、2 MiB、每模块8样本。分页最多200项/256 KiB。使用显式独占窗口和epoch/session映射，rank不取results数组下标。

[target_resolver.py](../../src/ascend_fi/inspect/target_resolver.py)依据真实args/kwargs与forward语义解析明确selector。当前可执行路径要求浮点FP16/BF16/FP32、连续strided二维、正维数、可证明局部分片特征维度；未知/meta/别名/不支持嵌套目标拒绝。shape只能证明输入形态，不能单靠[1,K]认定decode。

[plan_builder.py](../../src/ascend_fi/inspect/plan_builder.py)要求LIVE报告、完整枚举、当前可适配资格、唯一rank/epoch、TP对应worker数、PP=DP=1、EP=false、eager已证明。跨worker目标集合/selector必须可由同一owner表达；不回退first_float。精确转义并锚定module regex，禁止父子重叠、重复、超过512目标。

当前新计划支持REFERENCE_REQUANT、tensor_absmax、interval=1、有限倍率0<m≤20；不因此声明底层参考实现所有provider都已通过此入口。DRAFT/enabled=false，内容hash覆盖身份和策略。预算：

```text
max_hits = selected_module_count × (max_tokens + max_prompt_tokens + 16) × 2
max_hits ≤ 2^21
timeout_seconds = 3600
```

这是保守有限保护，不是精确scheduler步数。预算耗尽不自动清零；触顶不能声称全窗口。Qwen0.6B为112×(512+58+16)×2=131264。

[plan_executor.py](../../src/ascend_fi/inspect/plan_executor.py)在执行前fresh rebind/preflight；同一计划先通过normal→disabled→IDENTITY→recovery。receipt与当前执行器/计划/epoch关联，不凭磁盘identity.json自动授权。运行时[input_contract.py](../../src/ascend_fi/inspect/input_contract.py)复验实参，不仅启动时检查。

## 3. FakeRequant实际数学

源码：[fake_requant_reference_extension.py](../../experiments/fake_requant_reference/source/fake_requant_reference_extension.py)，`_scale_for`、`_requant_tensor`、`_apply_pre_hook`。

### Scale provider

| provider | 当前tensor实现 |
| --- | --- |
| tensor_absmax | input.detach().abs().amax()在原dtype求最大值，转FP32后÷127 |
| per_token_absmax | 最后一维abs().amax(..., keepdim=True)，转FP32后÷127 |
| fixed | config固定值转设备上FP32 tensor |
| module_runtime_scale | 指定module属性为tensor或数值时转FP32；未解析明确拒绝，不按同名scale猜语义 |

输入为空/非finite拒绝；base scale非finite拒绝。**tensor路径先clamp基础scale，再乘倍率**：

```text
s = clamp_min(base_scale, eps) × scale_multiplier
w = input.detach().to(FP32)
u = torch.round(w / s)          # 最近偶数舍入
q = clamp(u, -128, 127)
faulty = (q × s).to(input.dtype)
```

eps默认1e-12。输出shape/dtype保持，替换的是模块本次输入实参，原权重及原生量化参数不改。这不是packed INT4，不是W8A8转换。倍率1仍会Q/DQ，不能代替IDENTITY。

IDENTITY分支记录零改变并返回None，原args/kwargs原样向下游传递，不先计算Q/DQ再尝试恢复。

### 实现边界必须保留

- CPU辅助 `reference_requant_scalars` 用 `max(base×m, eps)`；tensor用 `max(base,eps)×m`。极小scale时可能不同。原5个CPU测试不证明此边界一致。本次不改算法，后续若修需独立回归。
- reference绝对误差先在**原tensor dtype**计算 `(faulty-input).abs()`，均值才转float；不能称为完整FP32误差。新增zero/L2诊断先将两边各转FP32再相减，是另一口径。
- 当前没有显式变换后isfinite gate；只校验输入/base和shape/dtype。参考transform的明确拒绝异常会跳过并记录；任意底层torch异常并非都被转为安全skip。不能宣称任意极值输入保证有限输出。
- 本次m20是已测配置，不是对未知模型/任意scale范围的安全性证明。

## 4. 计数、采样和请求生命周期

```text
install(disabled)
  → preflight/IDENTITY
  → each request: arm(reset counters)
  → selected input → transform → real module
  → finally disarm/remove
  → recovery and clean checks
```

入口hook_seen先计数，随后enabled/rank/module/layer/输入selector门禁；成功选择后candidate_seen增加，再判断interval/max_hits。successful transform或IDENTITY才增加hit。rank_filter拒绝不是目标rank未命中；非目标rank不应有hit。

max_hits是一次arm累计上限，达到后继续reject，**不是自动disarm**。每请求重arm才重置计数；不能在同请求触顶后清零变成无限all-hit。request_generation仅关联此次事务，不等于服务端排他锁。

[EffectStats](../../src/ascend_fi/inspect/effect_stats.py)累加所有接受调用，逐事件只保留首16/尾16。install/arm通过既有reset清空计数与统计；disarm保留测量。每模块接受调用1、65、129…采样zero/L2，编号从reset后重新开始，非随机、非全量。详见指标字典。

PrivateServiceOwner复用VerifiedOwner生命周期，增加显式deployment、私有鉴权、lease与限额。guard核对owner/result_dir/lease_id、参考SHA、≥12 GiB剩余空间、session≤4 GiB和events<20000。模型、HTTP鉴权、clean query与engine running/waiting零也需证明。

### 失败处理的真实边界（本轮只记录，不修复）

1. 请求异常后finally尝试disarm/remove并保留raw；HTTP超时**不等于引擎已经停止该请求**。若请求完成无法证明，写 `request_completion_unproven.json`，后续guard拒绝。不得删marker继续发请求，应先核实idle/worker和必要清理。
2. reference `fake_requant_remove`先清注册状态再逐handle.remove；失败返回 `remove_failures`。因此query显示uninstalled不保证物理hook全已移除。当前owner部分清理只检查RPC异常，不完整提升该字段为失败；后续需单独强化。不能在报告写“任何异常最终都恢复”。
3. starter获取lease后，部分准备发生在启动容器try块之前；这一阶段失败可能留锁。不得盲删NPU_LOCK，必须先查owner/lease/容器/进程，确认本任务无请求和资源才人工归还。
4. stop先核对本session lease与container label，Docker stop等待60秒后允许容器停止兜底。因此只能在当前请求/driver安全结束后执行，不是请求取消API。日志exclusive create使重复stop也可能失败，失败时保留记录核实，不绕过检查。
5. 以上是已知工程缺口，不由此次文档或离线测试自动解决；无人值守生产可靠性未认证。

## 5. 评价与恢复

[dataset_effect.py](../../src/ascend_fi/inspect/dataset_effect.py)保留相同suite/protocol的B/F/R、机制query和完整原文；[distortion](../../src/ascend_fi/evaluation/distortion/README.md)复用EvaluationResult，不另起schema。

先检查目标/rank/hit/changed/清理；无效fault不能作为“无效果”计分。轨迹分叉、确定规则告警、任务答案、预算终止分别记录。开放题无expected不计算Accuracy，length耗尽不自动算病态。恢复需clean/health，当前TargetPlan额外要求全文和token pieces精确一致。独立scheduler全窗口仍UNPROVEN。

本次证据只来自已有三题一个cycle，不重复NPU、不引入旧R1 PPL。指标分母与状态以字段字典为准。
