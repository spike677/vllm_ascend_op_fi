# npu_a5 原始警告及可观测性复核

本记录只读复核已完成的 a5；未重跑 NPU、未改原始数据或冻结分析器。

`B.log:30` 有一条原生 C++ 警告：`[Check][offset] Check input storage_offset[%ld] = 0 failed, result is untrustworthy17`。`F.log` 没有这条警告。

按合并日志的行序，警告位于 sequence 5 的 `after_direct_cpu_recheck`（B.log:29，view_created 后对 offset=17 的 view 读回已结束）与 sequence 8 的 `before_direct_cpu`（B.log:31，view_written 后对同一 view 开始读回）之间。冻结 `alias_probe.py:179–183` 表明这个区间跨越 `WriteView` / `view.copy_` 及其 msprobe 包装，还包含写入值和写后原张量的 offset=0 读回；这些 offset=0 调用没有该阶段标记。

因此，现有日志支持“警告在覆盖 copy_ 调用及其观测的区间内被输出”，不支持把它唯一归因到 copy_ 内核、msprobe hook 或某一次格式转换。标记仅包围非零 offset 读回，并非完整算子跟踪；C++ 日志时间与 Python epoch_ns 也未经统一时钟及输出顺序校准。不能沿用旧 a3 的“可能来自首次未匹配 view 读回”推断，更不能声称已经消除此警告。

本 case 的数值结论另有逐元素证据：健康 B 的 64 个目标元素正确写回，192 个非目标元素不变；F 的 view 与独立临时张量正确更新，但原张量 256 个值均保持初值；两端消费的是各自同一原张量，consumer 恰好有 64 个差异。全部 BF16 位模式、直接 CPU 读回、显式 ND 读回和读回前后身份检查相符。该结果限定在本几何、数值、格式和软件版本，不能推广为任意 offset 私有格式均可靠。

原生 statistics/mix 中，B 的 as_strided、copy_ 返回值及 CreateView 输出为 null 统计；F 对应 ND 输出可观测。null 与数值表示可见性差异，不能算数值错误，也不能证明两个输出相等。两端 consumer 模块输出统计均为 null。原生三件套独立发现的首个可对齐数值差异是写后共同观察器读取原张量（B `NPU.npu_format_cast.6.forward` / F `.8.forward`）；按共同语义模块输出比较，首差异是 `ReadOriginalBase`。这与写回丢失的传播位置一致；存储别名断开的结构位置由独立共同观察器证明，不能仅凭 statistics 唯一反推原生内核根因。

结论：a5 支持受控“视图别名断开 → 原张量写回丢失 → consumer 差异”的机理验收；警告原文和统计不可见性作为限制保留。它不构成自然发生的 CANN 缺陷复现或真实模型写回调用链验证。
