# Fault39 output_proxy：C3 累积注入与有限值门槛审查

本次只审查 source84 的失败证据和现行设计，不修改 Native、Provider、Runtime 或历史原件，不执行 SSH、容器或 NPU 调用。

**结论：现行“全部静态合格目标在同一请求内每次都注入”与 `finite-original` 门槛存在条件冲突。** 如果前序故障或其他原因使后续调用打开的原输出包含非有限值，后续调用必须拒绝。当前 C2 的独立请求成功并不保证累计 C3 成功。不能据此断言所有输入或所有模型都不可能完成 C3，也不能把本轮拒绝改记为通过。

## 本轮可以确认的事实

本地重新计算以下文件的 SHA256，读取了最终结果、失败回执字段提取和远端完整原件的独立审计回执。没有重新下载或检查私有完整张量；字段提取本身不是完整原件的替代品。

| 材料 | SHA256 |
| --- | --- |
| source84 最终 result | `80f3903b653c061d9a72630c4855237ba2c9e37ea93b018404b0f5c79b6c8a60` |
| 失败 Native receipt 字段提取 a1 | `303ae1aff51b19baac3de4495f562aa56fcbda16dcbf6b9d8c1c30d2452c19c6` |
| 独立完整原件审计 a2 | `f82934c4fc03cec6af780919fd74e7c2a3d02cbcdd95e6ba46472ad36370fa15` |

- 最终结果保留 `REFUSED_OR_FAILED`、`PostwriteRefused('POSTWRITE_APPLY_4')`。a2 审计为 106972 checks、无 issue，分类仍是 `FAILED_OR_PARTIAL_NOT_QUALIFIED`，整体完成为 false。
- a2 独立重建的已完成 C2：113 个完整 projection 调用位置中 28 个合格目标、28 个独立 fault forward、3277 条 ledger；其 C2 原件一致性成立。不是 113 个目标全部可注入。
- C3 的原 live C2 proof 发出与消费历史可以关联，但不等于 C3 完成。第一轮 `c3_fault` 被中断，后续恢复没有追认该轮成功。
- 唯一拒绝 Native receipt：BF16、shape `[14, 6144]`、86016 个元素、172032 字节，`mode=1`、`indices=[0]`。操作依次为 `open=0 → apply=4 → restore=0 → release=0 → live=false`，`changed_elements=0`、`closed=true`。
- 最终 owner 已释放；E5 仍是 `UNADAPTED_COLLECTOR_BOUNDARY`，本审查不授予统计定位结论。a2 的离线 C2/C3/E5 授权字段也均为 false。

源身份也有明确边界：本地 `postwrite.py`、`postwrite_coverage.py`、Worker group 和 provider 四个实现文件与 source84 结果中的哈希相同。审查的 `postwrite.cpp` SHA256 为 `c9068385487841b912fc21c2e14a348132070e1bc61f13a14016cd8cbe019b57`，与 compiler-source75 提交 `f800a488` 的对应 Git blob 完全一致。a2 已重哈实际 DSO `2e7f1bb9b957afa9ab1ef14da4aa802608d51234d736cefa93ecc765dde461d7`，但明确未重哈完整 build-fingerprint receipt 和外部模型权重。本审查不补造“源代码到该二进制的完整编译链已独立复核”结论。

## 哪些由代码证明，哪些仍不能证明

`finite-original` 是 [现行完整组契约](../../docs/worker_core_v1/output39_complete_set_contract.md#L51) 的明确约束。该文档也已说明传播导致的非有限值可以使 C3 拒绝。它不是为本次失败临时添加的条件。

根据上述已固定源码：

1. [`dcat_pw_open`](../../fault_injection/native_adapters/postwrite/postwrite.cpp#L101) 同步并保存本次实际输出 span 的原始 BF16 位模式。这里的“original”是**当前调用打开租约时的输出**，可能已受同一请求内前序故障影响；不是另一个健康 baseline 请求的输出。
2. [`dcat_pw_apply`](../../fault_injection/native_adapters/postwrite/postwrite.cpp#L120) 检查整个 `live->original` span。任何元素的指数全为 1，即 Inf 或 NaN，都会返回 4；并非仅检查待改写的索引。此分支在 fresh read、与原快照一致性比较、replacement 构造和写入之前。
3. [`mode=1`](../../fault_injection/native_adapters/postwrite/postwrite.cpp#L129) 的成功写入把选定 BF16 元素设为 `0x7f80`，即正无穷。当前 Python [proxy 配置](../../fault_injection/deployments/postwrite.py#L550) 明确使用这类 Inf/NaN 代理。它是受控原输出存储改写，不是证明真实 MatMul 内部溢出原因。
4. [组注入入口](../../fault_injection/deployments/postwrite_coverage.py#L230) 先按冻结目标集作 CoverageSession admission，再执行 Native apply。现行 every-match 要求所有静态合格调用都实际注入；不能在 apply 拒绝后自动缩小分母。[恢复](../../fault_injection/deployments/postwrite.py#L731) 还原各租约的原存储，不能自动消除已经计算出来的下游结果。

因此，**在已审查 Native 实现对应实际 DSO 的前提下**，return 4 表示打开的原快照至少含一个非有限 BF16 值，并且该次 apply 尚未执行写入。回执中的零 `changed_elements` 是 Python 初始化的输出参数仍为零，不能单独当作“逐元素比较完成”的证明。随后 restore 成功也只说明恢复到这个租约的原值，不说明恢复值一定有限；完整模型恢复要看另行执行的 OFF/health。

现有材料不能确定：

- 是 Inf 还是 NaN、正负号、数量、索引，或 index 0 本身是否已经非有限。
- 原快照的任何具体数值，或者 apply 瞬间整个 live tensor 是否仍与该快照相同；返回 4 时后面的 fresh-read 一致性检查尚未执行。
- 非有限值一定来自哪个前序注入、哪一个传播算子或自然计算。前序 Inf/NaN 传播是与代码相容的机制解释，但不能代替固定输入下的独立因果对照。
- 所有模型、输入或输出位置都必然发生同样拒绝。本轮的失败也不否定已完成的独立 C2。

## 保持机制语义的后续选择

| 选择 | 保留的语义 | 可以验收的范围 |
| --- | --- | --- |
| 保持当前固定全目标 C3，补因果观测 | 同请求全部合格目标累计注入 Inf/NaN，guard 不变 | 允许再次明确失败。只有每个目标实际 apply、consumer、两轮重复和 OFF/health 都完成，才能通过原 C3 |
| 每个目标在独立新请求中重复注入 | 同样的真实输出、代理值、consumer 和恢复；避免目标之间的前序污染 | 可证明每目标跨请求持续触发。必须单列“隔离重复”scope，不能替代原同请求累计 C3 |
| 新增条件 every-match scope | 对当前调用的整段原值满足有限值及原有条件时注入；不满足时保留原计算结果 | 可证明动态适用条件内的完整覆盖，必须新设计、新建原 live proof、新实机验收；不能继承或追认本轮 C3 |

若继续保留原固定 C3，最小补证是在**原调用、同一 Native 快照**上记录只读非有限计数/分类、完整 span 哈希、实际 producer/consumer/request 关系；对比健康、只注入前序目标以及原累计计划。需要把原快照证据与后续实际消费者连接起来，才能区分传播解释与其他来源。不能回放一个修改过的输入后再称为当前请求的原数组证据。可尝试另一个预注册输入或自然模型路径，但须保留完整目标分母和所有失败结果，不能只挑成功层。

## 条件 every-match 的必要边界

建议将它作为显式新 scope，而非修改现行“固定全目标 C3”的含义。静态目标仍由完整健康调用发现；动态适用性是每次真实调用的附加判定，不能从列表删除旧目标。

新实现和验收至少需要：

1. **Native 只读预检查。** 在本次真实输出和精确 owner/generation/storage/span 上，读取或复用同一受保护 Native 快照，记录整段 BF16 有限性、计数和哈希。Python `isfinite` 摘要、历史健康值或注入日志不能代替实际 Native 判定。预检查只能在已授权的资源预算内进行。
2. **保留 apply 的二次守卫。** 预检查通过后仍执行现有 finite-original、fresh-read、所有权、几何、并发写入及 readback 检查。预检查通过而实际 apply 拒绝，仍是执行失败；不能转成“条件不适用”或成功。
3. **完整分母与互斥分类。** 每次请求的每个静态合格调用都记录为“条件满足且实际注入完成”“预检查已证实原值非有限、无故障写入”“未知/冲突/失败”之一。静态不支持的调用也保留。三类必须覆盖原始分母，无重复、遗漏或失去关联；未知不得默认为不适用。
4. **不适用不是命中。** 对非有限原值不作清洗、替换或自动恢复到其他请求的 baseline。保留原生后续计算（可能继续传播非有限值或发生原生失败），并验证临时预检查资源释放。这个分支的计数不能补入 hit，也不能满足当前固定全目标 C3 的要求。
5. **不把来源作为已证事实。** “当前原值非有限”可由预检查证明；“因前序 F 传播而非有限”需额外因果对照。在证据不足时，只记录前者。还要区分原本就非有限的健康调用。
6. **新契约单独验收。** 两轮相同请求都独立核对动态分类和实际应用；至少存在可重复的成功注入，不能用全跳过的空结果授予执行能力。baseline/lossless/OFF、实际消费者、零遗留资源、完整分母和原件仍必需。新 C2/C3 必须绑定新的条件策略、源码、Native 构建和原 live proof，统计 E5 另行验收。

## 不应绕过的项目

- 不删除全 span 的 finite-original 检查，也不悄悄改为仅检查待写入元素。
- 不把 apply 4 改为返回 0、不吞异常、不把 `changed_elements=0` 标为成功命中。
- 不用 `nan_to_num`、clamp、写回健康 baseline 或改变原 Q/K/V、权重/下游输出来使后续目标重新有限。
- 不通过提前恢复、跳过真实消费者、只选最后一层或删去传播后不适用目标，使原固定全目标计划表面通过。
- 不把 `pos_inf` 换为零、符号翻转或小有限扰动后仍宣称同一 Fault39 Inf/NaN 代理机制；这些应另定变体。
- 不因正常恢复、C2 原件通过、原 live proof 已消费或 a2 审计一致，授予失败 C3 或 E5。

本轮建议保留“C2 原件一致、累计 C3 被 finite-original 约束拒绝且已恢复”的结论。若产品目标是“当前条件满足就持续触发”，优先明确设计上述条件 scope；若目标仍是固定全部目标同请求注入，则继续保留拒绝作为有效失败结果，不降低门槛。
