# Evaluation：机制、效果与恢复分开

## 职责、环境和边界

消费已验证目标的baseline/fault/recovery原始记录，通过共享EvaluationResult记录机制、任务/概率/输出、稳定性与恢复。不负责生产controller，不复制native或FakeRequant数学。规则、重放和报告可离线运行；live适配需要相应Linux私有服务。

## 输入 → 实现入口 → 输出

- [runner.py](runner.py)、[workflow.py](workflow.py)：已有ValidationResult和executor → 有清理的B/F/R → EvaluationResult。
- [metrics.py](metrics.py)、[signals.py](signals.py)：原始答案/token/logprob等 → 数字/信号；旧解析器的历史语义保留，不自动当默认V2。
- [live.py](live.py)、[suite.py](suite.py)：旧Qwen/native适配与suite；不等于新TargetPlan通用概率driver。
- [effect_profile.py](effect_profile.py)：配对结果聚合和Effect Profile；不虚构缺失PPL。
- [distortion](distortion/README.md)：当前确定性评分、完整原文、终止与病态区分、Dashboard。
- [audit.py](audit.py)、[replay.py](replay.py)：历史离线审计/重放，原raw不覆盖。

## 状态、失败处理和下游

缺失IDs/概率/性能保留null并记录basis；pieces不等于IDs。无效fault不能按“效果为零”计分；开放题没有expected不算Accuracy；恢复是有效性gate不是严重度。输出供report/profile消费，不作为下一次arm许可。

版本、采样和阈值以[指标字典](../../../docs/evaluation/fake_requant_tool_v1_metrics_zh.md)为准，生命周期以[算法](../../../docs/architecture/algorithms_zh.md)为准。[tests/evaluation](../../../tests/evaluation)是离线回归；本次Qwen0.6B三题N=1既有证据不证明通用PPL、DeepSeek或统计稳定性。
