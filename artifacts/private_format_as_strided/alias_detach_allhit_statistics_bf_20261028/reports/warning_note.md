# 故障 15：npu_allhit_a2 警告与统计可观测性复核

本记录只读复核已完成的真实 B/F 运行；未重跑 NPU，未修改 attempt 内的原始文件或冻结分析器。

B.log 和 F.log 均在第 30 行记录一条 `[Check][offset] ... result is untrustworthy17` 原生警告。两端第 29 行是 sequence 5 的 `after_direct_cpu_recheck`，第 31 行是 sequence 6 的 `before_direct_cpu`；相邻标记的 forward 都是 0，case_id 都是 `f0_guard_unregistered_base`。

冻结 `source/alias_probe.py:208–219` 中，该 guard 先观察未注册 NZ29 base 的 view，然后执行原生 `view.copy_(values)`，再观察写后 view。因而合并日志行序将这条警告限定在未注册 base 透传对照的写入及其观察区间，而非六个 eligible 注入目标中的某次命中。该 guard 在 B/F 中都保持原生别名，均精确写回 64 个目标元素且 consumer 正确。不能把该警告当作注入成功证据，也不能仅凭日志将其唯一归因到 copy_ 内核、msprobe hook 或某次内部格式转换。原生日志与 Python 阶段标记的时钟和输出顺序未经统一校准；一条日志也不能证明其他调用绝不会遇到相同内部限制。

两组各另有一条 `NPUCachingAllocator.cpp:202` 关于 CANN/HDK 需要按 32 padding 分配内存的原生警告。两份 runtime.json 各记录 14 条同类 `torch.jit.script_method` DeprecationWarning（torch/jit/_script.py:362）。这些记录均保留，不作为故障位置或命中次数的证据。

原生 statistics/mix 的可见性有边界：每次 B 的 NZ29 view 在 as_strided、copy_ 和 CreateView/WriteView 输出中四项统计均为 null；F 的对应 ND2 view 统计可见。null 对数值仅表示可见性不同，不能当成数值错误或相等。两端 ConsumeOriginalBase 模块输出也均为 null；consumer 的精确值与存储关系由共同运行观察器补充核验。

按原生调用栈和 construct 归属逐 case 对齐，step0/step1 的结果相同：

| call | 首个可对齐数值差异 API（B / F） | 首个语义模块差异 | 解释 |
|---|---|---|---|
| c0 | NPU.npu_format_cast.15.forward / .17.forward | invocations.0.read_original_base.ReadOriginalBase | 写后共同观察器先读到原 base 差异，随后语义读回模块可见 |
| c1 | NPU.npu_format_cast.24.forward / .29.forward | invocations.1.read_base_before.ReadBaseBefore | 新 view 创建前已有前次写回丢失留下的差异 |
| c2 | NPU.npu_format_cast.39.forward / .49.forward | invocations.2.read_base_before.ReadBaseBefore | 同上，继承已有差异，不能归为本次 as_strided 首次产生数值错误 |

每个 fresh base 的 c0 输入逻辑值相同；c1/c2 的 B/F 输入本来已不同。六次调用各自的 view 初值均忠实于该组当次输入，写后 view 均等于本次真实写入值。B 每次更新原 base 的 64 个目标，F 每次更新新的独立临时存储而原 base 不变；192 个非目标不变，consumer 每次恰好在 64 个目标位置有差异。六个 F 临时存储彼此独立，也与两个仍存活的注册 base 独立。

本结果支持固定 NZ29/BF16 几何及受控 Python 入口内的 all-eligible 机制验收。普通 msprobe 统计不能独立证明存储别名或唯一原生内核根因；它不证明自然模型写回链、历史 CANN 缺陷或生产事故已复现。警告和 null 限制不能因精确值检查通过而删除。
