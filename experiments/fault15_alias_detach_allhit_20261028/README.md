# 故障 15：每次满足条件均触发的私有格式视图别名断裂

**状态：本版已通过新的机制级 NPU B/F 实机验收。** 正式运行 `npu_allhit_a2` 在 124 物理卡 5、逻辑 rank 0 完成；健康组先通过 `PASS_BASELINE`，之后完整 B/F 获得 `PASS_ALL_ELIGIBLE_MECHANISM_AND_STATISTICS`。F 的 6 次符合条件调用全部触发，两组各 4 次 guard 均保持原生行为；逐次值与独立整数参考精确一致。控制器用时约 89.036 秒，自有容器已移除、锁已释放，结束检查为 `PASS_RUN_AND_CLEANUP`。这不是模型级或历史缺陷复现结论。

本机制的正确编号为**故障 15**。此前单次版本曾误标为故障 27：[历史单次实现](../fault27_alias_detach_20261028/README.md)的源码、原始采集及[完整证据包](../../artifacts/private_format_as_strided/alias_detach_statistics_bf_20261028/README.md)保留原路径、schema 编号和原始字节；公开说明文档另加编号纠错提示。本目录实现新的持续条件触发策略，不用旧单次结果替代本版验收。正式结果在独立的[本版证据目录](../../artifacts/private_format_as_strided/alias_detach_allhit_statistics_bf_20261028/README.md)交付。此前 a1 只在资源预检阶段被阻止，未执行本探针的 NPU 测试，不属于正式 B/F 数据。

## 故障因素与触发范围

正常的 `as_strided` 视图应与输入 base 共享存储，通过视图执行真实 `copy_` 后，原 base 对应位置也应更新。本方案在符合条件的 F 调用中，将传给原生 `as_strided` 的对象改为实际 `npu_format_cast(base, 2)` 返回的 ND 临时张量。几何参数保持不变。临时张量创建时应与该次原 base 数值相等，但存储独立；视图写入随后落到临时存储，继续读取原 base 的消费者看不到该写入。

| 条件 | 实际调用 | 写入后的预期 |
| --- | --- | --- |
| B：健康对照 | `original_as_strided(base, size, stride, offset)` | 视图与原 base 共享存储，原 base 更新 |
| F：满足全部选择条件 | `temporary = npu_format_cast(base, 2)`，再调用 `original_as_strided(temporary, size, stride, offset)` | 视图与 temporary 共享存储，原 base 不更新 |
| B/F：未匹配 | 对原输入执行原始调用 | 保持原生视图及写回行为 |

唯一注入因素是 F 分支中的实际格式转换。实现不使用 clone 回退，不重排数据，不改变 stride/offset，不替换 `copy_` 返回值，也不修改消费者。格式转换是否确实产生独立存储、是否保持创建时数值等价，由共同观察器和独立验收核实，不能只凭命中日志认定成功。

选择器同时绑定当前 PID、rank 0、活动中的真实 forward、指定 Python 调用位置、强引用注册的 base 对象及其完整存储身份、实际 NPU 格式和几何参数。base 必须为 format 29、BF16、shape `[16,16]`、stride `[16,1]`、offset 0；请求的 view 必须为 size `[8,8]`、stride `[16,1]`、offset 17。

策略为 `every_eligible_call`：没有命中次数上限，没有仅限 forward 0 的判断，也没有第二次匹配报错或首命中后禁用的逻辑。当前实验只证明这个受控 Python `torch.as_strided` 入口；不覆盖任意 Tensor 方法、ATen 入口或原生内部调用。实现仍有明确的 PID、对象和调用位置限制，不能把“每次满足条件”理解为对系统所有视图调用生效。

## 调用链与代码位置

```text
AliasMechanism.forward：选择本 forward 的原 base
  ├─ GuardOtherCallsite：已注册 base，不同调用位置，只读视图
  ├─ GuardUnregisteredBase：未注册 base，选中调用位置，真实写入与消费
  └─ InvocationBlock × 3：同一原 base 连续使用，不重置
       ReadBaseBefore(base) → ND 观测
       CreateView(base)
         → selected_as_strided
         → AliasInjector.dispatch
         → B 原输入 / F 实际 ND 临时张量
         → 原始 torch.as_strided
       WriteView(view, values) → 真实 view.copy_(values)
       ReadOriginalBase(base) → ND 观测
       ConsumeOriginalBase(base) → 真实 base + 1
```

| 位置 | 作用 |
| --- | --- |
| [alias_injector.py：注册与选择条件](alias_injector.py#L32) | 强引用两个 base 及其 storage，绑定 PID/调用位置，记录无 forward/hit 上限的策略 |
| [alias_injector.py：每次 dispatch](alias_injector.py#L79) | 对每次调用重新匹配条件；F 只增加格式转换，随后调用原函数一次；所有 F 临时张量保留强引用 |
| [alias_probe.py：两个调用入口](alias_probe.py#L52) | `selected_as_strided` 和不同调用位置的 `unmatched_as_strided` |
| [alias_probe.py：共同观察器](alias_probe.py#L76) | 同步记录原 base、view、实际临时张量的身份及小规模精确值；不读取 arm 或 hit 计数决定观测内容 |
| [alias_probe.py：语义模块与连续调用](alias_probe.py#L147) | 五个语义阶段、两种 guard、每个 forward 的三次真实写入与消费 |
| [alias_probe.py：构造输入与采集生命周期](alias_probe.py#L317) | 安装前创建两个注册 base 和两个 guard base，安装一次注入器，跨两次 start/stop/step 保持同一实例 |
| [build_reference.py](build_reference.py#L9) | 纯 Python 整数索引写入，生成各次独立参考，不使用 Torch/NPU/as_strided |
| [analyze_alias.py：独立参考重建](analyze_alias.py#L99) | 分析器自行重建索引、逐次写值和消费结果，不直接信任探针输出或生成器结论 |
| [analyze_alias.py：逐次状态与 guard 验收](analyze_alias.py#L200) | 验证每端自己的 pre/post、真实别名关系、实际 copy 来源与消费者；核验六个保留临时存储互斥 |
| [analyze_alias.py：原生覆盖与对齐](analyze_alias.py#L435) | 从 dump/stack/construct 独立检查每次 as_strided/copy_ 及语义模块，随后查找首个被观测数值差异 |

详细机器可读标准见 [acceptance_contract.json](setup/acceptance_contract.json)，具体整数参考见 [logical_reference.json](setup/logical_reference.json)。

`setup/acceptance_contract.json` 是运行前冻结的验收条件，设计状态字段仍保留 pending；实测状态以正式 [acceptance.json](../../artifacts/private_format_as_strided/alias_detach_allhit_statistics_bf_20261028/reports/acceptance.json) 为准。验收后不回写冻结 contract 的状态，以保持与 source_manifest 的原始 SHA 一致。

## 两次 forward、连续三次写入与参考结果

B 与 F 各用一个新进程。每组仅构造并安装一个注入器，跨两个真实 forward 保留它及累计计数。两个独立的注册 base 在安装前创建，初值都是 `arange(256).reshape(16,16)`，实际格式为 NZ29。forward 0 使用 base 0，forward 1 使用 base 1；每个 forward 内的三次调用连续使用同一个 base，不回填初值、不重新绑定 base。返回的 views 及 F 的六个临时张量一直保留，避免把存储地址复用误当作共享或独立证据。

目标位置为 `17 + i*16 + j`，其中 `i,j=0..7`，共 64 个；其余 192 个位置不应改变。三次 copy 来源分别为 `-64-j`、`-128-j`、`-192-j`，`j=0..63`。初值、写值与 `base+1` 的预期结果都能由 BF16 精确表示，参考比较不设置数值容差。

下面以原 base 的 `[1,1]`（线性位置 17）为例，**两个新 forward 的实测值均与以下独立参考完全一致**：

| call | 写入值（该位置） | B 写前 → 写后 | F 原 base 写前 → 写后 | B / F 消费结果 |
| --- | --- | --- | --- | --- |
| 0 | -64 | 17 → -64 | 17 → 17 | -63 / 18 |
| 1 | -128 | -64 → -128 | 17 → 17 | -127 / 18 |
| 2 | -192 | -128 → -192 | 17 → 17 | -191 / 18 |

对每次调用，验收要求 B 的 64 个目标位置相对自身前态全部改变，F 原 base 相对自身前态完全不变；两端实际写后 view 都等于同一 copy 来源，F temporary 的目标位置也随写入更新。消费者必须仍使用该 forward 的原 base，B/F 消费结果恰在 64 个目标位置不同，192 个非目标位置保持一致。

F 的每个 temporary 创建时必须等于该次原 base，并独立于两个已注册 base 和全部先前保留的 temporaries。仅比较每次临时张量与当前 base 不同还不够，分析器还核验六次临时存储彼此不同，防止复用并重置一个临时缓冲区也被当作六次实际转换。

每个 forward 在三次匹配调用前还执行两种 guard，因此每组共 4 次未匹配调用：

- 已注册 NZ29 base，完全相同的 view 几何，不同 Python 调用位置；只读 view 应与原 base 共享存储且不修改原值。
- 新建且未注册的 NZ29 base，从同一选中入口调用，执行真实 `copy_(-16-arange64)` 和原 base `+1`；两端都应保持原生别名及正确写回。

实测 B/F 各有 6 次匹配的原始调用、4 次未匹配透传；F 命中 6 次、B 命中 0 次。完整 B 两个 forward 先通过 `PASS_BASELINE`，之后控制器才运行 F。最终状态为 `PASS_ALL_ELIGIBLE_MECHANISM_AND_STATISTICS`；[joint 报告](../../artifacts/private_format_as_strided/alias_detach_allhit_statistics_bf_20261028/reports/acceptance.joint.json)还确认六个保留临时存储彼此独立，且不与任一注册 base 共享存储。

## 如何理解 msprobe 的首差异

对每个新 base 的 call 0，B/F 写前值应一致。结构性差异在创建 view 时发生：B view 指向原 base，F view 指向新的临时存储。两个 view 的初始数值仍应相同；执行同一个 copy 后，原 base 的状态才发生 B/F 差异。共同诊断读回中的格式转换 API 可能先观察到该差异；语义模块层则应在写后 `ReadOriginalBase` 观察到它。报告分别保留原始 API trace 和语义模块结果，不把诊断读回 API 自动称为故障发生算子。

同一 base 的 call 1、call 2 不同：之前的写入已经造成 B/F 原 base 的值不同，`ReadBaseBefore` 因而应观察到继承的差异。后续 CreateView 的输入和初始 view 值也可能不同，这不表示故障原因提前到新的输入阶段。每次持续注入的依据是该次新视图的存储关系，以及每端相对自身前态的完整转移；累计 hits、已有输出差异或重复出现的首差异，都不能单独证明后续调用确实再次触发。

a2 的[原生统计独立分析](../../artifacts/private_format_as_strided/alias_detach_allhit_statistics_bf_20261028/reports/acceptance.native.json)实际找到以下结果。表内原始 API 均为 `NPU.npu_format_cast.<编号>.forward`；编号只用于引用这次记录，对齐依据仍是所属 invocation、完整调用栈与父级。

| case | 首个被观测数值差异的语义阶段 | 首个对齐数值差异 API（B / F） | 解释 |
| --- | --- | --- | --- |
| f0_c0 | `ReadOriginalBase` | 15 / 17 | 共同观察器在 `view_written` 后读取原 base；本次新 base 写前无可用统计差异 |
| f0_c1 | `ReadBaseBefore` | 24 / 29 | 本次 view 创建前已观察到前次写入造成的差异 |
| f0_c2 | `ReadBaseBefore` | 39 / 49 | 本次 view 创建前已观察到前次写入造成的差异 |
| f1_c0 | `ReadOriginalBase` | 15 / 17 | 新的独立 base 重现首写后差异；共同观察器读回先于语义模块输出 |
| f1_c1 | `ReadBaseBefore` | 24 / 29 | 继承本 forward 前次写入的差异 |
| f1_c2 | `ReadBaseBefore` | 39 / 49 | 继承本 forward 前次写入的差异 |

每个 step 的 B dump 有 87 个节点、F 有 102 个节点；两个 step 共 B 174、F 204 个。原生覆盖门槛分别确认每个 step 的 5 次 `as_strided` 和 4 次 `copy_`（含 guard）。call 0 的共同读回最早差异为 Min `-127/0`、Mean `84.5/127.5`、Norm `2384/2352`；这是 msprobe 保存的统计值。精确逐元素关系由独立 joint 证据补充，不能仅从这些摘要反推存储身份。

本次两个 step 的同类原生 dump/stack/construct 字节相同：两次 forward 使用不同的仍存活 base，但采用相同初值、调用位置和写入序列，API 编号也按 step 重置。文件相同本身不能证明执行次数；实际两次 forward 由原始 start/stop 日志、completed_forwards 回执、不同活 base 身份及逐 forward 的真实调用事件共同核对，原始文件不做去重或重写。

分析分为相互独立的三部分：

1. **原生统计分析**只读取 config、dump、stack、construct，按 step、invocation 模块路径、construct 父级、API 类型及完整调用栈对齐；不读取 runtime、reference 或 control 来选择首差异。不能直接按不同运行中的全局 API 数字后缀配对。
2. **共同观察与整数参考分析**使用小规模完整值、BF16 位模式、object/TensorImpl/StorageImpl/数据指针、offset、实际格式等核验别名和每次状态转移。地址只在各自进程内比较；不跨 B/F 进程比较地址数值。
3. **控制与完成检查**核验每次 dispatch、6/4 次覆盖、两个真实 forward、采集回执及来源 SHA。计数不能替代前两项证据。

`null` 或非有限统计值表示不可观测，不能解释为相等，也不能单独当成数值故障。相同 Max/Min/Mean/Norm 不能证明逐元素相等，更不能证明共享存储。因此准确用语是“首个被观测数值差异”，其范围限于可对齐且可观测的节点，不能声称找到了全部原生执行中的绝对最早差异或唯一设备指令根因。

## 采集内容、运行与交付

配置为 `task=statistics`、`level=mix`、`summary_mode=statistics`、`tensor_list=[]`，逻辑 rank 0、step `[0,1]`；每个 step 对应一次完整微基准 forward，不是模型生成 token。两组使用相同源码和配置，只有 arm 与输出路径不同。

安装的普通 Python mix 支持列表未默认覆盖这两个 Python 入口，探针在 debugger.start 后使用官方 `register_custom_api` 注册 `Torch.as_strided` 和 `Tensor.copy_`，再安装注入器。所有原始调用仍真实执行一次；结束后逐个恢复注册。没有修改已安装的 msprobe 文件，也没有事后补造 dump。跨 step 如果官方采集生命周期意外替换已安装 API 绑定，探针立即失败，而不是继续产生不明确证据。

每组正式统计包包含 **8 个原始 JSON**：

```text
config.json
receipt.json
observation/step0/rank0/{dump,stack,construct}.json
observation/step1/rank0/{dump,stack,construct}.json
```

不生成 `.pt` 或其他张量二进制。单独的 `runtime.json` **包含这个小规模实验的完整逻辑值和 BF16 位值**，与只含统计摘要的 msprobe 数据分区保存；`control.json` 保存选择器与实际调用计数。不能把“没有 .pt”理解为所有交付文件都没有完整值。

实际执行由 [controller.py](controller.py) 先做资源检查，再运行健康组并独立验收，之后才运行故障组。它只清理本实验持有的容器；资源检查和冻结源码指纹也随本次证据保留。以下为已有运行目录的离线分析命令，不占用 NPU：

```bash
python analyze_alias.py \
  --baseline outputs/<accepted-attempt>/B \
  --fault outputs/<accepted-attempt>/F \
  --reference setup/logical_reference.json \
  --output outputs/<new-analysis-name>.json
```

输出路径必须全新。正式发布只接受控制器 PASS、健康先行门槛 PASS 和完整 B/F 独立验收通过；[tools/package_evidence.py](tools/package_evidence.py) 保留原生 JSON 字节与 receipt 引用，生成 B/F 各 8 个 JSON 的 ZIP、分区 runtime、报告、9 份冻结源码、来源映射及 SHA 清单。发布包的完整性校验可离线运行：

```bash
python package_evidence.py --verify .
```

这里的完整性 PASS 只表示交付文件及来源关联有效，不是一次新的 NPU 验收。本版 B/F 记录的实际版本均为 Torch `2.10.0`、torch-npu `2.10.0.post4`、mindstudio-probe `26.1.0`，两组源码指纹相同；版本与源码 SHA 取自本次 runtime 和冻结 manifest，不沿用历史单次运行作为本次证明。

## 警告与解释边界

共同观察器会同步读取 source，经显式 ND 转换取得逻辑值，并用直接 CPU 读回前后比对原对象身份与位值是否变化。这是测量手段，原生 trace 中会出现观察器自身的调用；它们必须与语义计算节点区分。

本次原始 B/F 日志各含 1 条 offset 17 的 `result is untrustworthy` 原生警告，以及 1 条 `NPUCachingAllocator` 的 32 padding 分配提示；各自 runtime 另记录 14 条相同类别的 `torch.jit.script_method` 弃用警告，不能宣称无警告运行。完整日志与 offset 读回阶段标记保留在证据包中。offset 警告两端均位于 `f0_guard_unregistered_base` 的写入及其观察区间，而不是选中目标的注入信号；该区间不能唯一分离设备内核、格式转换或统计 hook 的来源。逐次精确值、读回完整性和 guard 门槛均通过，但这不等于对任意 offset/格式的行为作保证。独立警告审查说明随本版证据包归档。

本实验通过了“视图错误地建立在独立等值临时存储上，真实写入无法传回原 base”机制的受控验证。结果只支持指定入口、固定几何和数据的机制级结论；不证明生产模型经过相同路径、不证明历史 CANN 缺陷自然存在，也不表示仅凭统计摘要便能唯一识别别名根因。本次通过依据包含每次原生调用覆盖、继承差异分类、独立临时存储、真实写回与消费者证据，累计 6 次命中只是其中一项控制检查。
