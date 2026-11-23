# r7 CPU 适配进度（2026-11-23）

本轮开始三批开发，不只更新计划。现有八项产品候选不变；Native 三项、DeepSeek 产品模式和 A3 33 仍未放行。以下 CPU transport、scheduler 和结构夹具不代表 NPU 或现役模型通过。没有启动新的 NPU 任务，也没有向黄区发消息。

## N：Native 41 → 26fia → 20

沿用 `AutomaticRequests → ProductRequest → ContinuousRequest.install_site`、原 Runtime owner/lease 和 `NativeRegions`，不新增状态机。补上 `product_native.request_evidence`，在产品请求 collect、释放 hook 之前检查每次事件属于当前内部 request ID、真实消费者字段、区域大小、原值/恢复 SHA、租约清零；26fia/20 额外核对复制来源与改写后 SHA。结果保留 Native 库 SHA 和逐区域证据，不用 hook 返回代替设备消费完成。

| 机制 | 现有真实边界 | 安全恢复点 | 本轮产品准备 |
| --- | --- | --- | --- |
| 41 | ScatterPaKvCache 写当前槽 → FIA 读取相同 KV 页/槽尾列 | 原 writer 完成后备份，FIA 同 stream 事件完成后恢复 | 产品 collect 保留 slots/page_mapping；prefill 非 paged 调用保留不适用原因 |
| 26fia | FIA 输出 → 当前活跃 residual → add_rms_norm | 消费者非原地、输出不别名旧 residual，同 stream 完成后恢复 | 校验 request、source/after、before/restored；不声称指针重定向 |
| 20 | 活跃 residual → DynamicQuant 输入 → 原 QMM | DynamicQuant 读完后恢复输入；QMM 消费独立 Q/scale，并记录设备完成 | 产品结果保留源/目的区 SHA 和消费者，不用输出扰动替代输入内容覆盖 |

CPU 验证使用真实 BF16 CPU 区域读写及矩阵消费者、显式 ACL transport 替身，真实复用 `AutomaticRequests`/`ProductRequest.collect`。三种机制分别检查两个 rank、同一组 runner 两个 owner、prefill+3 decode、OFF/health 新请求精确恢复。消费者必须实际读到替换值；pending lease、错误 request、缺少源 SHA 或恢复不符均拒绝。它证明共享生命周期接线，不证明 A3 的 kernel、stream 或 Native ABI。

尚未完成：黄区 Native 提取全文/源码/编译器/库身份审查；A3 安装 Python/source 的匹配；受信产品配置向 binding 传递已审查 Native 库；真实 HTTP 两 owner 与异常流恢复验收。`PRODUCT_CANDIDATES` 未增加三项，`PRODUCT_NATIVE_ADAPTER_PENDING` 保留。异常消费失败或对象版本改变时不得伪造 consumed、不得无条件写回，现有 owner 保留以拒绝继续运行。

## D：DeepSeek site 与动态请求归属

依据 `dsv4_0731_source_review_20261123.json` 的 exact Git blob SHA，候选 dispatcher 的 `token_combine` 调用 `npu_moe_distribute_combine_v2` 并返回输出；专家计数和 assist_info 是通信信息，不能直接当作用户请求行映射。V2-Lite 的 `MoE8Site` 是另一条 AllGather/unpermute 路线，不能沿用成 V4 EP 的资格。

新增 `dsv4_request_rows.combine_row_owners`：根据本轮真实 input_batch 请求顺序、scheduled counts、computed token 位置，以及适配层实际观察到的 output→scheduler row 映射做一一 join。拒绝重复/遗漏/越界行、未明示 padding、陈旧请求集合；没有调用的 rank 不制造调用。不在命中点通信，也不要求 fault 路由与 baseline 相同。

8 的 combine 输出行错位代理先形成 CPU 计划：只在同一请求已确认的输出行中循环移动，padding 不参与；单行请求保持不变，记录无可移动行。该计划尚不执行 tensor 改写，不修改 EP 协议或 assist_info。未来经消费者和租约审查后才允许将它连接到真实输出区域。

只观测工具修正：有请求列表且输出二维、行数大于 1 仍不能证明 combine 行归属，8 现在记录 `COMBINE_OUTPUT_ROW_JOIN_UNPROVEN`，保留 observed 数，不把它算作可注入调用。28/41 仍保留 sparse/compressed layout 未适配原因；Batch 只考虑独立 DynamicQuant，不替代融合量化。

剩余前置：正式噪声标定、全 rank 只观测、decode 重新计数、实际 loaded 源码和 padding/slicing 的运行时 join。只观测中的 typed candidate 不等于 site 已通过，CPU 构造映射不用于黄区组合登记。

## P：33 TP1 独立适配

保留现有 A2 TP2 的分组及公开门禁。TP1 不能使用 `TP2 参数 ×2` 或直接套 512/256：按其实际 q_heads/kv_heads、head_dim、KV 长度、block_size 和 launch 重新取证。TP2 每个 rank 也分别绑定，不用 rank0 代替全组。

新增只读 `tools/review_paged33_scope.py`：核对 observed/expected 的硬件、TP/rank、本地几何、runtime source SHA，再核对固定 QKV、launch/serializer SHA、spans、两次 native controls、精确重复性。逐位一致率按整数计数检查 `100*equal >= 99*elements`，不能仅信 `passed=true`。错误 TP、设备、launch SHA、几何元素数或重复不稳定均拒绝。输出永远 `qualification_granted=false`，不改变 registry；它只核对审查材料的关联，不能认证自报 scope 的真实性。

下一步：A3 TP2 的实际 launch/control 到达后再选分组；TP1 独立采集同类材料。满足控制后才推进 A3 验收和产品模式，两个 TP 的许可、命名变体和回传身份分开。

## 本轮 CPU 验证

使用既有 unittest 环境，20 个相关 unittest 模块按 test ID 去重后，97 项全部通过、0 跳过（r7 新适配、产品自动请求与 Batch、Native region/三 site/构建与前置门禁、DeepSeek observer/noise、33 binding/control、产品契约/审查及 r6 包门禁）。pytest 不在该环境中，没有安装新依赖，改用仓库现有 unittest。

无新黄区 r6 A 组回传；若到达先定性。性能不作为门禁，也不在本轮优化。r7 不冻结，未通过能力保持关闭。
