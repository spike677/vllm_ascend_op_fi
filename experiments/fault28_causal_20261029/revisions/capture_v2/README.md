# Capture v2

首轮 `stock_bridge_a2` 的真实 API 已返回，但 msprobe 递归注册跳过根 ReplayModule，导致统计中缺少 module。失败回执保留，正式矩阵有效 case 为零。

v2 仅用纯 `torch.nn.Module` 容器注册原 ReplayModule 子模块，供 msprobe 建立真实 module hook。容器不执行 forward，实际数值调用仍是原 module 的一次调用。采集回执继续要求 API 与真实 ReplayModule 父节点同时存在。

- 原实现和证据说明：`../capture_v1/`。
- 修改 diff：`../capture_v2.patch`。
- CPU 静态注册检查：`../capture_v2_cpu_review.json`，不等于实机验收。
- v2 六批 prepare-only 报告将保存到 `outputs/cpu_prepare/controller_plan_a2/`。

32 个 case、输入、vendor 和独立统计分析器不变。所有正式 case 均须使用相同 v2，不能混入 a2 的失败记录。新的 `stock_bridge_a3` 先确认真实 Module 采集及同输入基线，再继续后续矩阵。
