# 变更记录

## source-baseline-v1-20260911

- 冻结Qwen0.6B已验证实现：`f67b4186b7ed5652d49ac7c40aa9cc762adfeaf1`。当时live是 `b8b167d709271296b8114b42a627d3a7827d03b1` 加工作区修改，不能把事后commit号说成当时HEAD。
- 纳入固定参考源码/原测试、已有公开source suite，消除默认手册对旧E1/R1会话和其他工作树的隐式依赖。
- 添加相对单根VS Code工作区、中文分层导航、算法/产物/指标/生命周期合同。
- 新增离线文档与源码审计helper，不参与模型执行。
- 文档纠正tensor eps顺序和原dtype误差统计，未修改实现。
- 未改FakeRequant数学、生命周期、TargetPlan、EvaluationResult、阈值、native故障或旧证据；未启动NPU，未push。
- 旧main未提交Fault26修改单独保护，不混入本基线；保存去向和检查见发布记录。

未完成：scheduler级全窗口、通用TargetPlan概率事务、DeepSeek实机、异常清理强化、一键安装。
