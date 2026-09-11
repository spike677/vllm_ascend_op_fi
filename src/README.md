# 源码地图

执行平台是Linux／Ascend；合同、规则、评分、报告可离线检查。

```text
tools/private_demo_service.py         服务与资源lease
tools/run_model_target_plan_live.py   编排
tools/run_private_demo_once.py        宿主一次有损后停止自有实例（复用原stop）
    ascend_fi.inspect                识别、绑定、计划与输入合同
        experiment_extension         有界统计，委托固定参考算法
        experiments/fake_requant_reference/source
    ascend_fi.evaluation.distortion   B/F/R机制与确定性评分
    ascend_fi.evaluation.numerical_effect   采样能量派生（不改变换）
    ascend_fi.evaluation.detector_audit     冻结检测器回归／独立标签门禁
    ascend_fi.contracts               EvaluationResult等共享合同
```

详见[包职责](ascend_fi/README.md)、[算法](../docs/architecture/algorithms_zh.md)、[产物合同](../docs/reference/artifact_io_contracts_zh.md)。原生Fault20/26/39的C++仍在 `fault20_fast_validation/source/`，没有复制到Python包。

跨模型主要检查descriptor、角色适配、真实输入binder和部署owner，不另写FakeRequant或EvaluationResult。DeepSeek未实机验收，不能凭结构匹配进入有损执行。

新选择策略在 `inspect/selection.py`，安装／arm 全 worker 确认在 `worker_confirmation.py`。`single_effect.py` 仍返回既有 EvaluationResult，recovery=null，不把单次推理改名为完整周期。参考概率适配在 `evaluation/reference_probability.py`，现场概率事务尚未接通。
