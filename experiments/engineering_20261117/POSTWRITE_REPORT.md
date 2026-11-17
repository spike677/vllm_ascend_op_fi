# 39.output_proxy / 41.tail_column 构造机制验收

124 上这两条独立实现通过本轮构造 NPU 模块的自动选点、B/F/OFF、原存储恢复及
`statistics/mix` 独立首差复核。当前范围是 BF16、小规模普通格式张量、一个自动发现的
真实调用；不是预训练模型或公共 Worker 注入验收，不授予全匹配、持续触发资格。

| 机制 | 仅统计得到的首差 | 字段 | Baseline → Fault |
| --- | --- | --- | --- |
| `39.output_proxy` | `Torch.mm.0.forward/output/0` | Max | `8.0 → +Inf` |
| `41.tail_column` | `Tensor.index_copy_.0.forward/output/0` | Max | `4.0 → 2.0` |

独立比较使用相同配置和采集器、真实 `step0`、固定 `rtol=1e-5 / atol=1e-8`。
分析只读取原始 msprobe 三件套、配置与采集器身份，不读取 Native 命中日志或完整私有
张量。两组的调用栈和模块关系字节一致；B 自比没有可观测差异，完整记录结构可对齐。
39 的有限数变为正无穷属于非有限异常，不伪装成有限相对误差。41 的 Max 绝对变化为 2，
相对 baseline 为 50%。

`Mean`、`Norm` 的原始 null 完整保留：39 每组 10 项，41 每组 12 项。null 不当作相等。
首差前的可观测字段未先出现差异，但统计不能证明完整输入逐元素相等。比较器保留
`UNPROVEN_REQUEST_MAPPING`；这些是构造小模块，不能据此声称绑定真实调度请求。

实现和运行证据与上述首差分析分别核验：

- **39.output_proxy**：实际 `aten.mm.default` 先执行，随后在同一个 BF16 `[2,4]`
  输出存储的逻辑前缀元素 0 写入正无穷，真实 `aten.add.Tensor` 消费它。F 命中一次，
  消费者有 1 个元素变化，OFF 精确恢复，无待恢复 lease。这是明确的输出代理变体；
  它独立于 `39.matmul_range`，不宣称复现 Matmul 内部的范围计算错误。
- **41.tail_column**：真实 `index_copy_` 写模型持有的 `[3,4]` buffer 的 slot1，随后只将
  该 slot 最后两列（完整 buffer 逻辑索引 6、7）的有限数翻转符号，再由真实 add 消费。
  F 命中一次，2 个元素变化；非目标区域、原存储和 OFF 均精确恢复，无待恢复 lease。
  这是受控合法地址的尾列扰动；没有证明自然 vLLM KV writer 或原生越界写入。

初次采集把 `provider.run` 后的验收 `equal` 也录入，导致 B/F 结构不一致，该失败原件保留。
本次重新采集将窗口严格放在真实 `model(*args)` 周围，完整运行原算子、所属模块和消费者；
前置身份检查、结果精确比较、恢复验证位于窗口外。没有修改原始 JSON、比较阈值或过滤
模型算子来获得首差。窗口在模型、启动或同步异常时也停止采集。

39 的 HF Qwen3-0.6B 预训练模型试验被无损门禁拒绝。第三轮补充的实际调用诊断确认：
两次观察各执行 2,376 个原始算子调用，其中 `aten.linear` 197 次、`aten.matmul` 57 次，
没有出现当前 Provider 接受的 `aten.mm`。因此候选和已观察消费者均为 0，精确拒绝原因
为 `NO_ELIGIBLE_WRITER_WITH_OBSERVED_VALUE_CONSUMER`。没有执行故障、没有待恢复租约，
也不存在有效 B/F statistics 配对。这里不是模型结果已经证明不稳定，而是执行边界尚未
适配；不能将 `linear/matmul` 与构造用例的 `mm` 直接等同。见
[`postwrite_hf_refusal_summary.json`](postwrite_hf_refusal_summary.json)。

[`postwrite_constructed_summary.json`](postwrite_constructed_summary.json) 包含独立报告、
实际运行结果、原始三件套、配置、采集器、比较器和源码的 SHA256，以及全部 null 清单。
本次只提交实现和总结；新的统计原始数据、完整张量及构建二进制不随总结发布。
