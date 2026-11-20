# TND28 当前值关联：开发检查点

2026-11-20，本增量仍不授权 F，不代表 B3/B4/B5 通过。

原生连续化 child 完成且同 stream 同步后，在 Execute 内部受限 callback 窗口读取当前 Q、K、V 原存储、mask 和 prepared V。地址来自实际 typed API 和存活的 workspace，不接收调用者传入的设备地址。读取绑定原 pid/thread/token、一次 Execute、固定字节数，每个区域只读一次；窗口关闭后拒绝。

Python 只在内存中计算完整逻辑 SHA，按已核对的 stride/offset 从 V 原存储提取逻辑行；prepared V 必须等于当前公开 V，所有公开值还必须与本请求最初值相同。完整张量不发布。该观察不改输入/输出，不能替代 parent 参数和实际 launch 的关联。

- 配置：`PreparedInputCapture(..., check_live_values=True)`；默认关闭。
- ABI：`dcat_tnd28_value_read_v1`，归属和 SHA 必须与固定 private opapi 一致。
- CPU：69 项通过，含当前值改变、prepared V 不符、截断和错误存储范围；另有 Native callback/token/窗口/重复读取反例。
- 实际 SDK：CANN 8.5.1 生成单元编译通过，2026-11-20T11:36:29.654205Z 至 11:36:38.728138Z，NPU 调用 0。对象 SHA `c8fc5e59bddc7f8f36b419c7ba9eb02dcc9fd6f646850613f2eecab4e7bf97fd`。
- 当前仍缺：新 vendor 实际加载后的正常路径取证、Native 两阶段准入与实际 copy/launch 绑定，以及 B3/B4/B5。`TND_REPLAY_NATIVE_TWO_STAGE_ADMISSION_UNBOUND` 不得通过删除拒绝检查来消除。

组合限制：124 A2 / CANN 8.5.1 / 原生成 FIA 路径 / 固定已审查 TND 几何。CPU 编译不构成 NPU 或其他形状资格。

## 124 正常路径独立复核通过

实际源码 `88e5bfa7`，清单 `f796e135f0c7636492606d578d85ac1d3dad38d825b672ce65ffb55cad7ab6ca`，新 build `54fc75ce71fafaff41a8070692abe78382c8e7612a3c313f4c79ce5b77a88ae0`。A2 / CANN8.5.1 / vLLM-Ascend0.18 / Qwen3-0.6B FP16 / TP1 / 2048-token prefill / 卡2。

全部28层自然调用保留，只选择一个实际对象做原生观察。14条 typed 事件、10条研究 RPC、实际注册 ELF、Slice→FIA 参数对应及5份当前值完整哈希独立复核通过。当前 V 与连续化后的 V 哈希一致，Native read_mask=31，callback 已释放。真实 key 为 Slice 524000000、FIA 5000000000000200103；没有运行 replay key。

审查器明确区分10步研究协议与旧11步公共协议，旧协议检查不放宽。首次审查因旧 reader 未识别 execute_borrow 退出字段而拒绝；后继 reader 完整核对借用退出字段及反例，以同一原件重审通过。未改写运行数据。源码包缺少研究文件、控制器导入路径错误均在NPU运行前修正。

UTC 12:00:54.388316—12:02:58.175201，总窗口123.786914秒（含初始化及隔离）；28剩余预算13607.024709秒。health仍仅比较token；B3/B4/B5未通过，F仍关闭。B2依负责人决定按完整stock/private桥接通过，保留FUNCTION_BIND_NOT_UNIQUE限制；M128对照不再重跑。

抢占：12:00:54.579385起仅停止卡5的 `ptq_dcde1a8de6b13b34_list61-less-full-01_dev5_slot0_8d2aadea`，镜像 `cann:9.1.0-910b-torch2.10.0_npu-atk26.7.8`，12:00:55.622871返回0。卡2运行、卡5空闲。有界清理器12:02:58.175296结束并join；未修改派发器、Docker入口、驱动或CANN，未删除外部容器或数据。

[总结与原件SHA](../../experiments/worker_core_v11_20261118/tnd28_fresh_values_20261120_summary.json)。原始回执只保存在D盘和124。

## 后续只读 live-launch 接口（仅 CPU）

新增 `dcat_attention_observer_launch_count_v1` / `dcat_attention_observer_last_launch_v1`，仅返回本线程、当前 Native scope 中真实记录的 launch。结束 scope 即清空；未知 binary、失败 launch、记录失败、超限或异常会让整个窗口不可用，不回退到上一条成功记录。该接口不选择 key、不修改参数、不提供故障许可。

CANN8.5.1 无设备编译与局部状态机自测通过（失配token、未发生launch、小缓冲区、失败后禁止恢复、跨scope旧记录、字节/次数超限）。observer 源码 SHA `4713447df2dfc01dd223fe1c3481fd8891d85410398870b3d141b2a8aed5fe96`，库 SHA `e08f45d35302188f4d9131696d68d94f50c42a90c1c7d0af308a8a9d64d02c37`。此新observer未参与上面的NPU验收，不能套用其通过结果。下一步需将它与同次child调用前后窗口、两阶段Native门禁连接并实测。

## 两阶段 Native 候选增量（未做 F 实机验收）

新增 opt-in 同次 child launch 读回：原生 Contiguous 前后计数必须恰好增加一次；stage2 callback 在同步后校验当前 token/pid/tid、调用时窗、stream、成功状态及实际注册 Slice ELF。必须先完成五份当前值读取。默认关闭，旧数据不倒填为新证据。

host 使用版本化 `dcat_tnd28_replay_prepare_gate_v2`，只在原 GWS 内重新读取实际 parent/child 关系，允许准备 tiling，不表示 prepared 值已经存在。host 通过精确库路径以 RTLD_NOLOAD 借用已加载的局部 opapi，不加载或提升全局可见性。Execute 在原 child 返回并同步后进行第二阶段校验，且 parent 必须为准备阶段的同一个对象，最后一次性 CommitParent。任何校验失败仍走既有安全取消路径；不删除拒绝检查来伪造通过。

本增量仍未接到可公开调用的故障入口，F/B3/B4/B5 尚未验收。正常值、copy、host prepare、最终执行是分开的证据；CPU 夹具不构成实际 kernel 资格。


研究驱动现已独立接入 `TwoStageResearchProvider`，没有放宽公共 registry，也没有把 F 塞进 observe 阶段。宿主按同源研究 request_plan / next_phase 执行 observe、observe_repeat、baseline、baseline_repeat、fault、off、health；同一标准 Worker 重复两个不同 owner。基线重复、实际单目标 F、当前输入相同/Attention输出变化、OFF完整hidden/logits恢复均逐项检查。只有实际OFF通过才解除 needs_off；health先卸除故障API/Attention拦截，再安装三个只读root/logits观察hook。未完成/不确定恢复保留owner，不自动重跑F。

新研究状态与反例CPU测试7项通过，连同既有输入契约回归共24项通过；上述协议尚未在NPU运行，不登记B3/B4/B5通过。


## 两阶段真实 F 与跨作用域额度（13:24 UTC）

A2 / CANN8.5.1 / vLLM-Ascend0.18 / Qwen3-0.6B FP16 / TP1 / 卡2，a5 完成第一作用域全部七阶段。独立 reader 重读原始 RPC、typed 事件、当前值完整哈希、实际注册 ELF 和 launch 参数，确认同输入的 FIA 输出与 hidden/logits 改变，OFF/health 完整哈希恢复，资源清零。这仅覆盖一个作用域、一个实际自动选择目标，不是全层注入或公共入口资格。

第二作用域完成两次 observe，baseline 在分配 parent 前拒绝：Native Seen 进程账本限制4个 token，已被第一作用域的 B/B/F/OFF 用满。第二作用域没有执行 F，独立总报告保持 REJECTED_OR_INCOMPLETE。研究构建改为8次上限，不清空历史token，仍拒绝重复token和存活executor。必须新建构建清单并重跑两个作用域。

a1–a5 保守计入28窗口共1099.031777秒，剩余12507.992932秒。隔离均已结束，isolation_removed=true。五次原因分别为私有绑定、旧BNSD边界、copy ELF路径解析、cached bind离线解析、跨owner额度；保留失败原件。同一原因两次失败止损仍有效。后继JSON仅减少缩进，保留所有字段和数值，不修改已有原件。

[尝试及抢占隔离记录](../../experiments/worker_core_v11_20261118/tnd28_two_stage_attempts_20261120.json)。原始回执、完整张量不提交；部分独立审查原件留在D盘和124。
