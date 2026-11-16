# 当前阻塞

- **真实黄区宿主。** 在已核对的蓝区工作树和发布提交中，`src/demoncat` 不可用，目标 commit、实际 CLI、Catalog、RuntimeBinding、Controller 及 Worker RPC 签名待提供。蓝区文档合同和 CPU 模拟记录可离线审查。
- **新 Fault20 的独立验收。** CANN 8.5.1 SDK 构建通过。还需实际 Worker 加载确认、两个模型的单目标真实 Run、原 storage 和 post-close OFF。QMM 的 x1 与 WeightNz 的 self 各自验收。
- **Fault39 宿主验收。** 需要真实宿主传入一台实际 Worker、跨请求隔离及 replay/readback，验证单次原生命中和 OFF。独立包原有实机回执按原版本保留。
- **124 资源。** 2026-11-16 20:10（上海）只读门禁未发现卡 2/5 活跃进程或打开的设备句柄，也未观察到实验锁被持有。设备映射权限不等于活跃分配。真实设备试验前仍须重查 Pod/AscendJob、进程、设备句柄、容器和锁，按现场授权执行。
- **独立统计。** 385 个已发布原件哈希指向 124 私有目录。外部完整复算须取得相应只读权限。26 QMM rank1 step0 的 NaN 和 33 的 IFA 可观测性仍为缺口。
