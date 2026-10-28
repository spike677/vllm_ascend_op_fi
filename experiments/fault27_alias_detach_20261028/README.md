# 故障 15（历史单次版本，原目录误标 27）：私有格式视图别名断开与原张量更新丢失

> 编号更正：本目录的视图别名断开机制实际属于**故障 15**，此前误标为故障 27。本目录记录历史单次版本；当前每次满足条件均触发的实现与六次实机验证见 [故障 15 持续触发版本](../fault15_alias_detach_allhit_20261028/README.md)。历史源码和原始证据包保留原字节、路径及 schema，以下单次触发限制仅描述本历史版本。

**2026-10-28，124 的物理卡 5 上正式 B/F 已通过机制级与统计一致性验收：`PASS_MECHANISM_AND_STATISTICS`。** 健康 B 先通过独立门槛，随后 F 执行一次受控注入。实现、冻结源码和本次 [B/F 证据包](../../artifacts/private_format_as_strided/alias_detach_statistics_bf_20261028/README.md) 一并提供。这是构造的最小 NPU 机制实验，尚未证明真实模型写回调用链可达，也不表示历史 CANN 缺陷或生产事故已经复现。

## 故障如何实现

正常视图与原 base 共享存储，因此写入视图应更新原 base。F 在选中的视图构造入口，把原 base 转为具有独立存储的 ND 临时张量，再在临时张量上构造同一几何的真实视图。临时张量**创建时与原 base 逻辑等值**；随后视图写入只更新临时存储，继续使用原 base 的消费者得到旧值。

```python
# B：同一受控入口调用真实原实现。
view = original_as_strided(base, size=(8, 8), stride=(16, 1), storage_offset=17)

# F：唯一故障因素；格式转换与 as_strided 均实际在 NPU 路径执行。
etensor = torch_npu.npu_format_cast(base, 2)
view = original_as_strided(etensor, size=(8, 8), stride=(16, 1), storage_offset=17)

# 两组相同的后续计算，消费者始终收到原 base 对象。
view.copy_(new_values)
output = base + 1
```

这直接模拟“视图失去原存储别名 → 视图写入成功但原张量未更新 → 原张量消费者产生差异”。实现不重排 stride、不替换 `copy_` 返回值、不改写消费者输出，不把调用方的 base 重新绑定到临时张量，也不追加 clone 使转换结果满足预期。转换若改变初值、未产生独立存储或写坏非目标区域，验收会失败。

## 触发范围与次数

**当前实现是单次精确绑定，不是条件满足时持续注入。** 每个新进程只运行一个实际微基准 forward（逻辑 rank0、forward0、step0）。选择器同时绑定 PID、固定 Python 调用位置、仍存活的 base 对象及 StorageImpl、实际 device/format/dtype/shape/stride/storage offset，以及请求的 view 几何。

B 在选中位置调用一次原实现，注入命中数为 0；F 在相同位置调用一次原实现，注入命中数恰为 1。B/F 的同一选中调用若再次完整匹配，都会报 `Selected call occurred more than once`，不会再次注入或静默透传；实验模块的第二次 forward 也被拒绝。每组另有一次同几何、不同调用位置的未匹配调用，原生透传与正确别名均已核验。

覆盖仅限这个受控 `torch.as_strided` Python 入口。它不意味着已经拦截 `Tensor.as_strided`、直接 ATen 调用或内部 native 调用；接入模型前需要单独证明真实路径经过该入口。

## 固定几何与环境

| 项目 | 本次实际设置 |
|---|---|
| 设备与软件 | Ascend 910B4，CANN 9.1 环境，torch 2.10.0，torch-npu 2.10.0.post4，mindstudio-probe 26.1.0 |
| 原 base | BF16，实际 FRACTAL_NZ / format29，shape `[16,16]`，stride `[16,1]`，offset 0，初值 `arange(256)` |
| view | size `[8,8]`，stride `[16,1]`，storage offset 17 |
| F 临时存储 | 实际 ND / format2，创建时等值，与原 base 存储独立 |
| 写入值 | `-64-arange(64)`，排列为 `[8,8]` |
| 消费者 | 真实 NPU `original_base + 1` |
| 目标索引 | `17 + i*16 + j`，`0 ≤ i,j < 8`，共 64 个不重叠位置 |
| 非目标区域 | 其余 192 个位置 |

[CPU 逻辑参考](build_reference.py) 只使用整数索引，不调用 Torch、视图或 NPU 格式，也不参与故障与消费者路径。所有初值、写入值和消费者结果都在 BF16 可精确表示的连续整数范围内；逐元素验收不使用数值容差。

## 实际调用链与代码位置

```text
AliasMechanism.forward（一个真实微基准 step0/rank0）
  → 同步观察原 NZ29 base
  → 未匹配调用位置：真实 as_strided，验证原生透传
  → CreateView → selected_as_strided → 精确选择器
      B：原 base → 官方 API 观察包装 → 原 as_strided
      F：npu_format_cast(base, 2) → 同一观察包装 → 原 as_strided
  → WriteView → 官方 API 观察包装 → view.copy_(new_values)
  → 共同观察器记录写后 view、原 base、临时存储
  → ReadOriginalBase：原 base 的 ND 读回副本（明确的观察节点）
  → ConsumeOriginalBase：真实 original_base + 1
```

| 位置 | 作用 |
|---|---|
| [alias_injector.py:32](alias_injector.py#L32) | 保留 base/存储强引用，绑定实际身份与几何 |
| [alias_injector.py:73](alias_injector.py#L73) | 精确匹配、未匹配透传、第二次匹配拒绝 |
| [alias_injector.py:99](alias_injector.py#L99) | 唯一故障转换；随后实际调用原 `as_strided` |
| [alias_probe.py:52](alias_probe.py#L52) | 被绑定的真实 Python 调用位置 |
| [alias_probe.py:142](alias_probe.py#L142) | CreateView、WriteView、ReadOriginalBase、ConsumeOriginalBase 语义模块 |
| [alias_probe.py:71](alias_probe.py#L71) | 共同观察器：同步、存储身份、完整小规模值及读回前后校验 |
| [alias_probe.py:297](alias_probe.py#L297) | 官方 msprobe 扩展 API 采集与逐项恢复 |
| [analyze_alias.py:197](analyze_alias.py#L197) | 共同运行证据与独立逻辑参考的逐元素/别名验收 |
| [analyze_alias.py:394](analyze_alias.py#L394) | 仅读原生统计、调用栈和模块结构的首差异分析 |
| [analyze_alias.py:438](analyze_alias.py#L438) | 单独核验命中次数、透传和采集完成回执 |

[controller.py](controller.py) 在受控单卡容器内先运行 B，并要求健康独立验收通过后才执行 F；源文件在运行前冻结并校验。本次完成后的资源记录确认自有容器移除、设备锁释放。

## 本次结果

B 的 view 与原 base 共享存储。F 的 view 与独立 ND 临时张量共享存储，与原 NZ29 base 不共享。身份比较针对同一进程中仍存活的对象与存储，不比较 B/F 两个进程的裸地址。

| 检查 | 健康 B | 故障 F |
|---|---|---|
| 写前 view | 与独立初值参考一致 | 与同一参考一致 |
| `copy_` 后 view | 64 个值全部写入成功 | 64 个值全部写入成功 |
| 原 base 的目标 64 个元素 | 全部正确更新 | 全部保留旧值 |
| 原 base 的非目标 192 个元素 | 全部不变 | 全部不变 |
| `view[0,0]`（写后） | -64 | -64 |
| `base[1,1]`（写后） | -64 | 17 |
| `output[1,1]` | -63 | 18 |
| 消费者差异 | B/F 差异索引恰好为全部 64 个目标位置 | 同左 |
| 注入命中 / 未匹配透传 | 0 / 1 | 1 / 1 |

[完整验收](../../artifacts/private_format_as_strided/alias_detach_statistics_bf_20261028/reports/acceptance.json)为 `PASS_MECHANISM_AND_STATISTICS`；[共同运行证据验收](../../artifacts/private_format_as_strided/alias_detach_statistics_bf_20261028/reports/acceptance.joint.json)确认结构差异首先被观测于 `view_created`，原 base 的数值差异首先被观测于 `view_written`。F 临时张量在写入后也被更新，不能将“创建时等值”误读成全程保持初值。

## msprobe 采集与首个被观测数值差异

两组均为 `task=statistics`、`level=mix`、`summary_mode=statistics`、`tensor_list=[]`。每组保存原字节配置、完成回执及 step0/rank0 的 `dump.json`、`stack.json`、`construct.json`。主统计 ZIP 各只有这 5 个 JSON，没有 `.pt`；独立 `runtime/{B,F}/runtime.json` 仍包含本次小规模张量的**完整逻辑值和 BF16 位值**，与统计证据分区存放。

此安装版本默认 mix Python API 列表未覆盖 `torch.as_strided` 和 `Tensor.copy_`。正式运行通过官方 `PrecisionDebugger.register_custom_api` 补充观察这两个真实入口，结束时逐项恢复原函数；没有修改安装文件或重写 dump。原生证据实际覆盖 `CreateView → Torch.as_strided.1.forward`、`WriteView → Tensor.copy_.0.forward`，未匹配调用为 `Torch.as_strided.0.forward`。B/F 原生 dump 分别有 23/28 个节点，F 的额外转换与临时对象观察不按全局序号强行配对。

[仅读原生数据的分析](../../artifacts/private_format_as_strided/alias_detach_statistics_bf_20261028/reports/acceptance.native.json)给出两种明确范围的结果：

| 对齐范围 | 首个被观测数值差异 |
|---|---|
| 包含共同观察器的可对齐 API | B `NPU.npu_format_cast.6.forward` / F `.8.forward`，调用栈均为写后 `view_written` 阶段读取原 base |
| 共同语义叶模块输出 | `Module.read_original_base.ReadOriginalBase.forward.0` |

前一对读回输出的 Min 为 **-127 / 0**，Mean 为 **84.5 / 127.5**，Norm 为 **2384 / 2352**（B / F）。这与“视图写入成功、原张量写回丢失”的传播位置一致。它是**首个被观测数值差异**，不是声称此前所有未配对或不可观测节点均没有差异。

机制异常发生在视图构造时的别名断开；数值差异需要经过写入后读取原 base 才显现。不能要求两者必须位于同一 API，也不能仅凭统计值唯一反推某条原生内核指令的根因。结构位置由共同观察器的存储身份与真实写入证据验证，原生统计分析不读取注入成功日志来指定首差异。

## 警告与结论限制

B 保留了一次原生 offset17 `result is untrustworthy` 警告，F 没有该警告。同步阶段标记将日志位置限定在覆盖 `copy_` 及其观测的区间内；现有记录无法唯一归因到 copy_ 内核、msprobe hook 或某次格式转换，也未证明警告已被消除。原文、标记范围及限制见 [警告审查说明](../../artifacts/private_format_as_strided/alias_detach_statistics_bf_20261028/reports/warning_note.md)。

B 的 as_strided、copy_ 返回值与 CreateView 输出统计为 null，F 对应 ND 输出可观测；两端消费者模块输出统计均为 null。null 只表示不可观测，不能判相等或判数值错误；四项统计相同也不能替代逐元素验证。消费者的 64 个差异由独立共同观察器确认。

本次健康 B 的逐元素写回、非目标区域、消费者、直接 CPU/显式 ND 读回及源对象前后身份检查均通过；这些证据只覆盖上述几何、数值、格式与软件版本，不能推广为任意非零 offset 私有视图都可靠。尚未实现或验证自然模型中的“私有格式 → 视图原地写入 → 原 base 消费”完整路径。

## 获取与校验证据

从仓库根目录执行以下命令，无需 Torch 或 NPU：

```bash
python artifacts/private_format_as_strided/alias_detach_statistics_bf_20261028/package_evidence.py \
  --verify artifacts/private_format_as_strided/alias_detach_statistics_bf_20261028
```

该命令验证文件、来源映射和两组统计 ZIP 的 SHA 与字节一致性；`PASS_PACKAGE_INTEGRITY` 是证据完整性结论，不是重新执行 NPU 验收。完整目录分区、历史路径映射和冻结源码见 [证据包 README](../../artifacts/private_format_as_strided/alias_detach_statistics_bf_20261028/README.md)。
