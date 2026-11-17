# 机制定义与证据合同

每个标准卡包含 Fault ID、Mechanism ID、名称、家族、Root Cause、Simulated Semantics、Injection Primitive、注入层、Operator、Model Path、触发、参数、可观察效果、恢复、证据和限制。

历史真实根因需要原始研究材料证明；当前模拟语义描述本次改变了什么；Primitive 描述具体执行动作。例如 Fault22 的返回行替换不能证明 uint32 地址溢出，Fault17 的 Conv3d padding 参数变更不能证明物理 Kernel 缺陷。Fault39 output proxy 与 matmul range 分开，FA 与 FIA 分开，FIA legal tiling 与 denominator replay 分开。

注册能力、CPU 契约、Native 构建、模型实机验收、C2/C3 是独立维度。本轮不提升历史 C2/C3、模型质量或多目标实机覆盖。每条继承证据保持原模型、CANN、SoC、TP、源提交和适用范围。

统一会话保存独立 B/F/OFF 实际调用观察，恢复结果为 R；重复阶段按列表保留。实际 Tensor 结果保留在本地会话，控制回执不冒充 msprobe dump。CPU verify 的精确输出比较不能替代目标 NPU 的完整 dump.json/stack.json/construct.json、配置、collector 身份和独立统计。需要实机验收时继续复用 engineering_closure.evidence 的哈希和统计审计。

源码回执列出真实文件 SHA256；生成矩阵中的 historical 字段不会因统一接口测试通过而变化。恢复失败时保留原资源与错误，不制造 OFF 通过；cleanup 被拒绝直到原 owner/generation 恢复成功。没有 NPU 的验证回执必须记录 npu_calls=0 和 qualification_granted=false。
