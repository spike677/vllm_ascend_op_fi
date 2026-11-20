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
