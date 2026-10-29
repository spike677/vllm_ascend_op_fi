# Fault28 任务本地 high 统计独立推断

两步独立分析都把**首个被观测统计差异**定位到 `Functional.scaled_dot_product_attention.1.forward` 输出，序号 105、父级 `model.layers.1.self_attn`。输入 Q/K/V 的 12 项有限统计匹配，输出 Mean 首先出现差异：B=`0.0002203190088039264`，F=`0.00022031953267287463`。

[step0.json](step0.json) 与 [step1.json](step1.json) 由同一冻结分析器直接读取 config/dump/stack/construct 生成；不读运行时哈希、host profile 或命中信息来决定首差。1621 个节点完整对齐，28 个 SDPA 的 28 层归属完整，未排除任何诊断节点。首差前仍有 50 项不可观测字段；这项结论是观测边界，不能宣称找到了硬件最先产生误差的指令。SDPA 是 PyTorch API 边界，原生 dump 未直接记录 CANN 内部 kernel。

此结果依赖明确披露的 **task-local msprobe 26.1.0 FP32 high collector 补丁**，不属于未修改官方默认统计模式。原官方默认 a2 的 [独立分析](../msprobe_inference_a2/README.md) 首差仍在下游 `Tensor.pow.7`，失败记录没有被覆盖。四项聚合统计即使提高精度，仍不能证明任意两个张量逐元素相同。

[formal_runtime_crosscheck_pinned.json](formal_runtime_crosscheck_pinned.json) 是另行执行的值透明性验证，没有参与上面的首差推断。它将正式 B/F 各两 forward 的全部目标层 Q/K/V/attention 输出、layer1 完整 native 输入与 logits 对照此前 OFF/ON 诊断，结果 `PASS_RUNTIME_CROSSCHECK`。只允许已审查的诊断探针 `06c783…` 与 high 探针 `aecb9c…`、helper `9f9d69…`、collector `3aa7ed…`、FA vendor `63a517…` 组合；逐一核验冻结 source manifest，原生模型源码与运行参数严格相等。报告明确不声称两版采集探针源码相同。

两步三件套及推断报告分别字节一致；两次实际 forward 的执行需要 receipt、实际调用和独立 callback 证据共同确认，不能仅由两个相同 JSON 路径推出。这里不替代主流程最终机制验收，也不宣称复现了历史模型故障。
