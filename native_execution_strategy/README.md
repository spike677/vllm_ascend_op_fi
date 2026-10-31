# 算子执行策略注入（第一期）

`native_execution_strategy` 把调用上下文、逐次选择、版本解析、资源预算和证据解释统一到一个接口。
第一个适配器是 `FlashAttentionScore/S1S2`；MatMul、Reduce/Softmax 和 FIA 仍待适配。
运行结果分别记录 `strategy_switched`、`numerics_observed`、`model_impact`，未知项保持 `null`。
合法策略切换不保证输出发生差异，也不等同于历史缺陷已复现。

## 范围与版本

本轮验证记录见[124 实验](../experiments/native_execution_strategy_20261030/README.md)。
当前构建已通过20 case/56调用的正式实机矩阵，`version: validated` 解析到 `cann9.1-s1s2-v1`。
证书 SHA 为 `be58b4e77098a8911ab97ad8711144a5a5999dc3742080726c594ad88e8fcc5b`。
原始 experimental 配方仍独立保留；新源码或新构建必须重新验收，不能继承本次状态。
证书是版本和证据绑定清单，不是数字签名或跨版本安全保证。

2026-10-31的[完整模型扩展](../experiments/fault28_model_coverage_20261030/README.md)
在同一构建上通过：HF Qwen3-0.6B、B1/B8共560次SDPA，符合条件的112/112次受控prefill
全部真实切换，覆盖28层及B8全部8个请求；504个layer/request映射对无遗漏。
280次范围外decode保留于总分母并原API透传，受控轮占112次。重复fault稳定、关闭后完整恢复，
独立msprobe首差在第0层SDPA输出Mean，差异传播至全部请求logits。
此结果仍受下列版本/形状范围约束，未扩展证书。
[原始statistics/mix与审计子集](../artifacts/native_execution_strategy/fa_model_coverage_statistics_20261031/README.md)
不包含完整私有张量或整段DEBUG日志。

FA 当前范围是 Ascend 910B4、CANN 9.1 `V100R001C11SPC001B243`，
源码 `c77c0829f1f0f0b8f882db1d9b051a758b605472` 的 S1S2 分支，FP16 BNSD，
Q heads=16、KV heads=8、S1=S2=2048、D=128、causal sparse mode=2、keep probability=1。
受控候选只覆盖实机验收后的 B1/B8 和 M∈{64,128}、nRatio∈{4,8}；B2/B4 原生发现结果
不能自动授予受控支持。候选还绑定完整输入属性、34项公共输入字段，以及166项已解码执行字段
和三个 SDK 区段的完整原始字节（共169项）。这三个区段尚未展开成员语义，不当作 padding 忽略。

`validated` 必须唯一解析到具体适配器版本、源码清单 SHA、私有 vendor 清单 SHA、证书 SHA、
当前 Python 源码 SHA 和实测报告。改变代码、SDK、构建或未覆盖形状会使校验失败。
未知调用默认拒绝；`on_unsupported: passthrough` 只允许真实调用上下文不支持时透传，
不会吞掉证书、版本、指纹或证据错误。

## 配置

使用 JSON，或安装 PyYAML 后使用安全 YAML 加载。配置拒绝未知字段、重复 key 和非有限数。
用户描述选择目标，不提供 tiling raw 的偏移或内部字段：

```yaml
mechanism: native_execution_strategy
adapter:
  operator: FlashAttentionScore
  version: validated
  route: S1S2
  on_unsupported: reject
selection:
  target: query_tile
  policy: legal_alternative
trigger:
  mode: every_match
  condition: adaptive
discovery:
  workload_axis: request_count
  max_cases: 32
  resource_budget:
    max_elapsed_seconds: 300
observation:
  verify_native_launch: true
  compare_complete_target: true
  msprobe:
    task: statistics
    level: mix
    summary_mode: statistics
```

FA 的 `selection.target` 支持 `query_tile`、`key_group`、`joint`。候选由适配器按证书过滤，
通用 Policy 从中确定性地选择与当前原生策略不同的候选。可在 `trigger.predicates` 中按
`request_count`、`token_count`、`seq_lens.max`、`matrix_dims.*` 等语义字段加条件；
未知语义值保留 unknown，不擅自用 `shape[0]` 替代。当前 BNSD 的 B 由 FA 适配器解释，
TND、Linear 的合并 token 数和 MoE 专家工作量需要各自适配器提供语义。

`every_match` 每次求值，没有默认命中次数限制。`condition: adaptive` 表示按实际上下文和
有证据的候选选择；它不代表自动寻找某个必然产生误差的 Batch 阈值。停用时使用
`trigger.enabled: false`。`selection.adaptive_rules` 可配置有序语义条件和候选 ID 偏好，
未匹配且无显式后备偏好时保持原生路径。

## 调用链与真实切换位置

```text
actual_call → adapter.inspect → InvocationContext
           → existing native observation → Policy.evaluate
           → adapter-certified alternative → adapter.native_scope
           → original API (exactly once)
           → actual host + launch + public POD evidence → independent observation
```

[`StrategySession.execute`](session.py) 为 native、controlled 和停用路径提供同一个原调用位置。
受控执行前必须已有同上下文的实际 native 观测；框架不会隐式追加 baseline 调用。
调用方提供原 API 的闭包和只读取证 observer；Session 原样返回原 API 的对象，不替换、
不重试、不修改输出。observer 失败保留原返回值和显式错误，验收不能把未知项判为通过。
需要严格比较 msprobe 完整 stack 的调用方应从同一源码行调用 `execute`。

FA Python 适配器通过作用域内 `NES_FA_PLAN_JSON`/`NES_FA_LOG_PATH` 把类型化计划传给私有 host。
作用域绑定 PID/native TID、调用 token、实际输入、精度属性和构建证据，退出后恢复环境；
并发或嵌套作用域被拒绝。它不是跨进程通用全局开关。

实际切换在 [`native/fa_bridge.h`](native/fa_bridge.h) 的 host 适配路径，
于原生 `CalcNRatio` 结束后原子选择 M/nRatio，随后重新执行依赖它们的 split、UB、
core、matmul/softmax tiling 和 workspace 生成。不会在已生成依赖数据后只改两个 raw 字段。
原有 kernel 模板和精度属性保持一致，候选必须满足该分支的已验约束。
详细构建与源码顺序证据见 [native/BUILD.md](native/BUILD.md) 和
[native/CONTRACT.md](native/CONTRACT.md)。

实际 launch 的 raw、key、block 数和 workspace 必须与本次公共 POD/host 回调相符；
配置、命中日志或期待的 M/nRatio 不能代替实际执行证据。
[`adapters/fa_schema.py`](adapters/fa_schema.py) 解码全部205项字段并保留原始1464字节，
仅在语义比较中排除已做源码审计的5项 padding（17字节），不删除原始证据。
205个字段覆盖1272字节；另以公开 section 边界完整绑定192字节的 softmax/transpose 区段，
两者合计覆盖全部1464字节。未展开区段采用逐字节比较，不能据此推断其中每一字节的含义。
审计依据见 [padding_source_audit.json](notes/padding_source_audit.json)。

## 原生发现与受控切换

[`discovery.py`](discovery.py) 接受语义调用和观测回调，按预算收集原生策略；
不假设 Batch 与策略变化单调。相邻区间补测由调用方给出明确 case，不能将未测点插值为支持。
完整执行计划与 M/nRatio 选择器分别比较：选择器相同也可能出现其他 native core 划分变化。
原生策略未发生变化同样是有效结果。

命令行是只读检查和已采集记录分析入口，不自动创建容器或调度硬件：

```bash
python -m native_execution_strategy validate-config native_execution_strategy/examples/fa_query_tile.json
python -m native_execution_strategy adapters
python -m native_execution_strategy discover --config config.json --records observations.json
```

`plan` 必须提供 SHA 绑定的实际 native 记录，只输出 `INTENT_ONLY_NO_EXECUTION`。
`discover --records` 不执行新 NPU 调用；新实测使用实验 controller 的资源锁、逐次设备门禁、
时限和内存限制。Python 预算不能抢占一个已经阻塞的原生调用，外部 controller 负责强制时限。
缺资源测量时保留 unknown，不用0冒充空闲或低消耗。

## 取证与扩展

msprobe 继续使用 `task=statistics/level=mix/summary_mode=statistics`，保留 dump、stack、
construct、配置和完成回执。统计首差只从这些数据独立推断；实际策略验证、完整目标输出
精确比较、模型传播分别取证。完整私有输出用于逐元素验算，不上传为主 statistics 数据。
本轮使用已披露的本地 FP32 high collector 修正，不能宣称是未修改的官方采集器。

新增适配器应实现 [`protocol.py`](protocol.py) 的调用识别、原生策略观测、合法候选、
候选校验、真实原生作用域及实际执行证据解释，并注册明确版本和证据。
某些分支需要匹配的新 kernel 模板和 tiling key，不能通过放宽当前 FA 白名单直接支持。
模型接入还必须证明包装覆盖真实调用、目标请求行映射正确及后续消费者传播；
算子 replay 的通过不能填为模型级通过。
