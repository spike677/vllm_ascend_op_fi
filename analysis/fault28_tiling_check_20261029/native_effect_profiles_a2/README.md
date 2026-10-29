# Fault28 a2 原生 FA tiling 元数据

`profiles/` 包含 5 组已完成运行的 252 次实际 FA launch：OFF B1 28 次、OFF B8 56 次、ON B1 guard 56 次、ON B8 56 次、ON→OFF recovery 56 次。

509 个文件共 2,424,680 字节，逐文件 SHA 与来源映射见 [profiles_manifest.json](profiles_manifest.json)。每个调用包含日志中的 1472 字节 aligned 内容与按原生日志声明长度保留的 1464 字节 prefix，以及每组原始行摘录/调用关联的 `profile.json`。**这些 `.bin` 是 tiling 元数据，不是模型张量、KV cache 或权重。** 不包含原始大型日志或 `private_replay` 数据。

提取器保留原 `CAPTURED_UNDECODED_FRAMING_CANDIDATE` 状态，没有改写成通过。选中层的 public-context raw 与实际日志 prefix 的逐字节相等、105 字段 POD 解码和控制条件，由 [独立 effect 报告](../native_fa_effect_a2_report.json) 验证。其它层的 raw 范围与 OFF 对照也在该报告逐调用比较。日志窗口关联和 CPU 源码推导不能冒充设备内部执行轨迹。
