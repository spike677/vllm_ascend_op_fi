# 实现架构

[算法与状态逻辑](algorithms_zh.md)是识别、绑定、FakeRequant、生命周期的唯一详细实现说明。[源码地图](../../src/README.md)定位文件，[产物合同](../reference/artifact_io_contracts_zh.md)定位数据，[指标字典](../evaluation/fake_requant_tool_v1_metrics_zh.md)定义计算口径。

本次基线保留现有源码布局，不重构native，不新增controller，不将观察环境（如msprobe）包装成故障后端。
