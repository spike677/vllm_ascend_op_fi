# DeepSeek／DPSK：下一步迁移，未实机认证

当前模型适配包含只读结构规则，不代表原生DeepSeek已完成真实输入绑定、IDENTITY或有损验收。Qwen的输入维度、TP分片、融合命名、epoch与目标数均不能直接迁移。

沿用现有入口：descriptor → adapter角色 → 真实观察 → binder → TargetPlan → 同epoch IDENTITY → B/F/R → 原EvaluationResult。先核对dense/MoE与专家并行、融合模块和局部特征维度。未知形态返回blockers，不降级为first_float或强开Qwen资格。

可能适配文件：[model_descriptor](../../../src/ascend_fi/inspect/model_descriptor.py)、[model_adapters](../../../src/ascend_fi/inspect/model_adapters.py)、[target_resolver](../../../src/ascend_fi/inspect/target_resolver.py)、部署owner。复用参考数学、统计和评价，不新增“DeepSeek fault backend”来绕过门禁。

当前状态：PENDING_INSTANCE／READONLY_ONLY；本轮没有启动DeepSeek或NPU。
