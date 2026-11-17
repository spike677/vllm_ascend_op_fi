# Fault26 residual_stomp：真实 Worker 的单调用验收

124 的预训练 DeepSeek-V2-Lite 标准 vLLM 推理链通过本轮
`PASS_COMMON_WRITER26_BFOFF`。它完成了真实模型自动选点、一次实际 QMM 写入干预、
后续 residual 消费、完整 logits 数值传播和 OFF 恢复，达到 **C1 单个自然 writer** 的机制范围。
**原始 statistics/mix 的独立定位未通过 E5**；不能用 Native 命中记录替代该项验收。

本轮为 TP1、单请求、14-token fresh prefill；每个阶段只生成 1 个 token。
vLLM 为 `0.18.0+empty`，vllm-ascend 为 `0.18.0`。标准启动流程创建临时 Worker；
Extension 没有创建 Worker，也没有替换 `runner.execute_model` 或 `_prepare_inputs`。
真实结果保留原阶段：step0 reference、step1 lossless B、step2 comparison F、step3 OFF。
这些 step 是四次独立 prefill，不是四个 decode token。

## 自动选点与实际机制

注入入口按实际模型对象、Native QMM 调用、仍存活的 residual、allocator 的两个相邻活动
分配块及下一次真实消费者进行匹配，不要求用户填写层名。本轮观察到的所属位置是
`model.layers.4.mlp.shared_experts.down_proj`；它是结果，不是选择条件。
实际算子为 `aclnnQuantMatmulWeightNz`，输出及 residual 均为 `[14, 2048]`、57344 字节。
B/F 的模型对象、模块对象、算子、cache、几何和 allocator 证据一致。

实际执行链为：

```text
真实 AddRmsNormBias 产生并保留 residual
→ 实际 QMM 自动匹配相邻 output/residual 分配
→ F 将 QMM 目标地址向前移动 544 字节
→ 真实 QMM 写入跨过 512 字节 padding，覆盖 residual 的末尾 32 字节
→ 将正确 QMM 结果放回原输出并恢复 padding
→ 下一次真实 residual 消费者读取已被覆盖的 residual
→ 消费完成后恢复原 residual，并读回验证
```

这是一项明确的 **QMM output-base underflow 受控代理机制**，不是历史 allocator 或
kernel 缺陷自然发生的证明，也没有在模型结束后修改 logits。

| 检查项 | 实际结果 |
| --- | --- |
| B 的匹配/claim/完成 | 1 / 0 / 0；正常透传执行 |
| F 的匹配/claim/完成 | 1 / 1 / 1；完成事件与独立 writer 原件一致 |
| 写入/消费者序号 | 225 → 226；consumer 返回状态 0 |
| residual 被覆盖范围 | 32 字节，其中 31 字节实际改变 |
| 正常输出恢复 | shifted-output 与原输出 hash 一致，原输出读回精确恢复 |
| padding/residual 恢复 | 写前与恢复后 hash 相同，精确读回检查通过 |
| 生命周期结束 | inflight、lease、pending consumer、queued executor、prepared 均为 0 |
| 移除 | finish、unregister、detach、unset 完成；无待恢复状态 |

公共 Provider 的预算是一个实际合格 writer。Native 内部的 `all_hits` 字段不代表公共
`every_match` 或所有层/所有请求已覆盖。本轮不授予 C3，也不扩大到其他形状、模型或版本。

## 模型数值传播与恢复

四阶段真实输入的完整 hash 相同。独立 model pre-hook 对实际输入字节计算 hash；调度器
CPU metadata 只作请求映射佐证，不被当作设备输入相等的证明。正常 reference、lossless B
及 OFF 的完整 hidden/logits hash 均一致；F 的两项 hash 均改变。返回 logits 是完整
`[1, 102400]`，四阶段都通过 finite 检查。

| 完整张量 | Reference = B = OFF 的 SHA256 | F 的 SHA256 |
| --- | --- | --- |
| hidden | `58cd625a1d8c888d646f64053817da7939bb157739b04c3ada0dbd48ef8ac7c8` | `3acf69f5263c76934d5f74a07de38ff0e6d8e7fe082387f43792f756fadd758e` |
| logits | `7f0b59420ae5f51c1ffc6c47cdb18bfc6fd91fa46ad1660c75ce806606d6f262` | `428ce67fe410089c6be71d2515d2bbefdaade08bb98310a6fd854413ac640287` |

Top-1 和实际生成 token 都是 `[185]`，文本都是一个换行。结论是完整 logits 数值发生
变化且 OFF 精确恢复；不能宣称生成文本改变、持续 decode 正常或任务准确率已经验收。
完整私有张量没有作为独立统计定位的输入，也不随本次总结提交。

## statistics/mix 独立结果：未通过定位验收

比较只使用原始 B/F `dump.json`、`stack.json`、`construct.json`、配置和采集器身份，
固定 `rtol=1e-5`、`atol=1e-8`，baseline 为参考。比较器没有读取 Native 日志或完整张量。
B/F 采用相同配置、相同采集器，调用栈和模块关系的原始字节 hash 相同。

独立首个可观测候选是：

```text
index:     86
operator:  NPU.npu_kv_rmsnorm_rope_cache.0.forward
argument:  input_args/4
field:     /Max
B → F:     61 → 77
```

实际安装的 MLA 调用源码把 `slots.to(torch.int64)` 放在位置参数 4。B 的新请求使用物理
slot 48–61，F 的新请求使用 64–77，正好解释这个差异。它是请求之间的物理页调度状态
变化，发生在选中 QMM 之前，**不是本轮 residual 覆盖的独立首差证据**。
不能删除这一字段、用运行日志指定更晚的首差，或将相同 token 输入等同于全部 API 输入
相同。独立比较器仍保留 `UNPROVEN_REQUEST_MAPPING`。

另外，B 自比在 `Tensor.__setitem__.11.forward/input_args/0` 的 `/Max` 遇到
`NaN / NaN`，报告 `NONFINITE_UNCOMPARABLE`，不能当作相等。完整原始 dump 中，
B/F 各有 10142 个 null 字段，非有限字段分别有 24 和 28 个；全量清单保留在原始数据和
独立报告中。因此 `baseline_self_comparable=false`、`whole_dump_comparable=false`。
这里没有修改统计字节、降低阈值、排除算子或将 null/NaN 转换为正常值。

下一次要完成 E5，需先获得有独立依据的请求/缓存元数据对齐和可比采集条件，重新收集
未被这些更早差异遮蔽的 B/F。现有机制成功与这个定位缺口分别保留。

## 实现与可复核材料

- [`residual26.py`](../../fault_injection/deployments/residual26.py)：实际模型 owner、保留
  residual、allocator 邻接条件、类型化 Native ABI 及单次预算。
- [`serving26.py`](../../src/demoncat/runtime/vllm_ascend/worker_extension/serving26.py)：标准
  Worker 请求映射、实际输入/完整 logits、B/F/OFF 阶段及恢复门禁。
- [`fault26_qmm_output_underflow.cpp`](../../fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_qmm_output_underflow.cpp)：
  真实 writer 地址切换、消费者关联和原存储恢复。
- [`writer26_worker_probe.py`](writer26_worker_probe.py)：实际部署与真实 enqueue 入口。
- [`residual26_worker_summary.json`](residual26_worker_summary.json)：原始结果路径/SHA256、
  构建 manifest 和 library 身份、实际阶段 hash、Native 状态、原始统计三件套/配置/采集器/
  比较器 hash、独立定位失败与台账 scope。Native 证据和统计比较分开记录。

本轮只发布实现和总结。新的原始统计、Native 二进制及完整私有张量留在 124；
既有机制证据和其他 scope 不覆盖、不合并为本轮通过结果。
