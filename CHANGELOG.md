# 变更记录

## fake-requant-usability-v1（2026-09-11，离线候选）

- 默认手选交集，显式 all_bindable／参与 ranks all；中文目标预览和精确绑定集合。缺 worker／异构集合／不可证明 rank 映射直接拒绝。
- 在新选择计划执行前确认所有 worker 的身份、安装和 arm；不改参考 Q/DQ 算法，不在 hook 中增加 collective。
- 新宿主单次有损入口复用现有容器停止流程；保留旧 B/F/R。实例停止、热恢复、重启后正常对照分别命名。
- 从已有有界采样能量派生 NMSE/SQNR；检测回归和独立标签质量验证分开。原检测阈值、scorer、历史证据保持不变。
- 可选 D：严格单窗口固定参考 token echo 解析及 NLL/PPL，仅离线，未连接完整 TargetPlan 概率事务。
- 不占 NPU、未做多 rank 现场验证、未新增 fault、不做敏感搜索、不 push。详见本轮发布记录。

## source-baseline-v1-20260911

- 冻结Qwen0.6B已验证实现：`f67b4186b7ed5652d49ac7c40aa9cc762adfeaf1`。当时live是 `b8b167d709271296b8114b42a627d3a7827d03b1` 加工作区修改，不能把事后commit号说成当时HEAD。
- 纳入固定参考源码/原测试、已有公开source suite，消除默认手册对旧E1/R1会话和其他工作树的隐式依赖。
- 添加相对单根VS Code工作区、中文分层导航、算法/产物/指标/生命周期合同。
- 新增离线文档与源码审计helper，不参与模型执行。
- 文档纠正tensor eps顺序和原dtype误差统计，未修改实现。
- 未改FakeRequant数学、生命周期、TargetPlan、EvaluationResult、阈值、native故障或旧证据；未启动NPU，未push。
- 旧main未提交Fault26修改单独保护，不混入本基线；保存去向和检查见发布记录。

未完成：scheduler级全窗口、通用TargetPlan概率事务、DeepSeek实机、异常清理强化、一键安装。
