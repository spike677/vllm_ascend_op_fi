# Inspect、TargetPlan与受控执行

## 职责与执行位置

从真实结构和入口tensor元数据生成精确目标计划，再委托已有owner做IDENTITY和B/F/R。Inspect不改权重，不按shape猜module→native对应关系。规则可离线测试，真实观察在Linux vLLM worker；组合扩展严格校验[固定参考依赖](../../../experiments/fake_requant_reference/README.md)。

## 实际接口：输入 → 输出

- [extension.py](extension.py)：`demoncat_inspect`白名单action/args → 分页结构或观察事件。
- [model_descriptor.py](model_descriptor.py)、[model_adapters.py](model_adapters.py)：config/真实类/runtime → 身份与角色候选。
- [session.py](session.py)、[target_resolver.py](target_resolver.py)：同epoch独占窗口实参 → selector、dtype/layout/局部分片维、blockers。
- [plan_builder.py](plan_builder.py)：绑定报告＋角色＋参数 → DRAFT/disabled的 `FAKE_REQUANT_TARGET_PLAN_V1`。
- [plan_executor.py](plan_executor.py)：计划＋fresh reports＋owner → IDENTITY和B/F/R；[input_contract.py](input_contract.py)每次调用复验。
- [experiment_extension.py](experiment_extension.py)、[effect_stats.py](effect_stats.py)：原算法＋有界聚合，不是新fault backend。
- [dataset_effect.py](dataset_effect.py)：固定数据集pair → 现有EvaluationResult。
- [client.py](client.py)、[private_demo_auth.py](private_demo_auth.py)：客户端和私有演示HTTP鉴权。

## 状态、失效与下游

epoch、runtime fingerprint、reference hash、目标集合和输入合同共同限定计划；新模型/服务重新观察。旧identity文件不是执行许可，执行器持有的receipt与本次验证关联。缺实参、未知布局、别名、父子重叠或过期身份拒绝，不静默删目标。

观察限时限量，停止核对handles；注入finally委托owner清理。`request_generation`只是关联号，不能自动隔离并发。异常限制见[算法](../../../docs/architecture/algorithms_zh.md)，字段见[产物合同](../../../docs/reference/artifact_io_contracts_zh.md)。

[测试](../../../tests/runtime_inspect)覆盖规则/合同/替身控制流；实机证据见[本次发布](../../../docs/releases/source_baseline_v1/README.md)。DeepSeek、独立scheduler窗口和通用TargetPlan概率事务尚未验收。
