# Fault39：固定全目标 C3 的一次拒绝诊断

本次保留原 `39.output_proxy` 的固定目标集、`pos_inf`、同请求 every-match、finite-original 守卫及恢复顺序。不会实现 conditional every-match，也不会把 apply 4 改为成功。此前设计审查中讨论的条件 scope **不在本轮实施范围**。

目标只有一个：区分 source84 的 `POSTWRITE_APPLY_4` 是否与同一请求内前序 F 传播相符。source84 原失败不变；新诊断即使支持传播解释，也不授予 C3 或 E5。

## 固定历史位置和本轮预算

历史 source84 的拒绝回执来自 `model.layers.1.mlp.gate_up_proj`，实际 API 为 `vllm.unquantized_gemm.default` 的 ordinal 0，完整 BF16 shape `[14, 6144]`、86016 元素、172032 字节。其前面唯一已注入的合格目标为第0层的 `mlp.gate_up_proj`。依据保留的原结果 SHA256 `80f3903b653c061d9a72630c4855237ba2c9e37ea93b018404b0f5c79b6c8a60` 与失败字段提取 SHA256 `303ae1aff51b19baac3de4495f562aa56fcbda16dcbf6b9d8c1c30d2452c19c6`，这些名称仅用于限制诊断位置，不成为故障注入选择接口。

新进程仍须从两个真实健康 inventory 解析这两个**当前对象**；固定 113 个完整调用、28 个合格目标，前两个合格位置及拒绝 shape 均须匹配。不能复用旧进程指针或旧 live proof。若模型、输入、实际调用顺序、eligibility 或 shape 不同，在 group lossless 前停止本诊断，不扩搜索。

利用原公共协议中自然存在的三个臂，不增加新的模型 forward：

| 诊断臂 | 原协议请求 | 额外观测 |
|---|---|---|
| 健康 | `c2_lossless` | 第1层目标返回后、原 admission 之前，现有 `read_span` 完整 fresh copy |
| 单前序 F | 首个 `c2_fault`，仅第0层目标注入 | 第0层实际 apply 后复制同一 lease.original；第1层目标返回后只做 fresh copy |
| 原累计 F | 首个 `c3_fault`，原固定全部目标 every-match | 第0层和第1层实际 apply 返回后分别复制各自同一 lease.original |

额外诊断最多 **5 条 compact 观测**，只涉及这两个位置。原 C2 所需的28次独立目标请求仍完整执行，其余位置不加额外 snapshot。预算为：6 个原前置请求 + 31 个完整 C2 请求 + `c3_lossless` + 首个 `c3_fault` + 最多2个恢复 OFF/health + 1个注销后健康请求，最多 **42 个协议及注销后请求**；若沿用 probe 的安装前健康请求，总计最多 **43 个用户请求**。vLLM 初始化本身的原生 warmup 单列，不算故障命中。

只做一个新 attempt。首次累计请求在预注册位置返回4即停止 F，执行原公共 recover/OFF/health/unregister/detach。若预期位置实际 apply 成功，sidecar 在该真实返回后发出明确的 `FROZEN_NATIVE_REFUSAL_NOT_REPRODUCED` 诊断停止异常，保留实际 Native return 0；它不伪造 return 4，不继续累计搜索，也不完成第二轮 C3。其他 Native 失败保留原错误；对齐、观测或恢复缺口均停止并报告未闭合。

## 最小实现及不变部分

- [`postwrite.h`](../../fault_injection/native_adapters/postwrite/postwrite.h) 与 [`postwrite.cpp`](../../fault_injection/native_adapters/postwrite/postwrite.cpp) 只增加 ABI-1 `dcat_pw_copy_original_v1`。在现有 mutex 与 handle/owner/generation/thread 校验下，复制该租约的 `original`，不会 fresh-read 设备、调用 ACL、更新 `original/expected/applied` 或创建租约。拒绝输出缓冲区与任意活动租约的 tensor/original/expected 重叠。输出是完整逻辑 span 的 BF16 小端位模式和64字节 metadata。
- 原 `open/apply/restore/release/live/read_span` 函数体保持不变。CPU 合同固定它们的源码摘要；尤其保留 `finite-original → fresh read → 相等检查 → replacement/write → readback` 的顺序。return 4 位于 fresh read 之前，其 `changed_elements=0` 仍不能当作逐元素比较结论。
- [`output39_refusal_diagnostic.py`](output39_refusal_diagnostic.py) 在实验 sidecar 中接入；不修改 Core、provider、注册表、production wrapper、hook 顺序或消息格式。`copy_lease_original` 核对实际 open/apply 回执、精确 tensor 对象/span/地址/device/owner/generation 和返回的 applied 状态，然后计算全 span SHA256、有限值/正负 Inf/qNaN/sNaN 数量及首个非有限索引、位模式和符号。不序列化完整数组。
- 真实 public direct-discovery 已占用 `sys.setprofile`，本实现不替换或串联它。sidecar 只在原 `_NativeLease._native_call`、`ProjectionCoverage.projection`、`end_observed_forward` 的精确 code object 上使用独立 `sys.settrace`，关闭 line/opcode 事件，拒绝已有 trace。源码路径/哈希与实际 loaded code identity 另存。仅移除自己拥有的 trace；原 Core profile 由原异常恢复路径清理。
- 两个 control 的来源明确为 `ACTUAL_FRESH_READ_SPAN`；累计拒绝位置必须为 `SAME_NATIVE_LEASE_ORIGINAL`。二者不能互相冒充。每次记录读时点、实际 request/forward、producer proof、当前对象和存储几何。consumer 只复制原 dispatch 实际记录的值读取；失败 apply 后未执行的 consumer 明确记为 `NOT_REACHED_AFTER_ACTUAL_APPLY_REFUSAL`。
- sidecar 仅弱引用候选对象，在每个原 forward 结束时复制 consumer 元数据，不延长张量分配的生命周期。它不添加临时诊断租约、不恢复到其他请求的健康值、不清洗 NaN/Inf、不提前恢复前序 F。

新 export 必须重新从本次确切 Native 源码编译并保存编译器、SDK、image、构建指纹和 DSO SHA。旧 source84 DSO 没有该接口，拒绝回退到 fresh copy。运行时检查实际映射文件的 inode/hash，并要求 copy 符号地址位于该 DSO 的可执行映射。此地址校验不能替代独立构建来源审核。

## 可执行实验入口

[`output39_refusal_probe.py`](output39_refusal_probe.py) 直接组合原 `natural39_worker_core_probe.drive`，不复制或重写公共请求状态机。`BoundedDiagnosticClient` 在原6个前置请求的最后一次 `collect_replay` 成功后，通过现有 callable `collective_rpc` 在**同一 Worker 原线程**安装 sidecar，核对真实公共 RPC ticket、内部 provider ticket 及其不同 owner 的关系。它不增加 Core action、registry 或新的 WorkerExtension。

首次累计请求后保留原 Native 失败，再调用原 public abort/recover/OFF/health/unregister/detach。关闭观测的错误不能阻止 Native 恢复；若新证据文件写入失败，保留该错误及待重试的实验 metadata，不宣称实验资源全部释放。诊断原件独占创建，不覆盖旧文件。正常关闭后移除自己的 Worker 属性和 trace，另执行一次普通请求核对 token/text；原 public recovery health 已独立检查完整 hidden/logits。

直接容器内入口如下；参数必须来自本次新构建，不得填写历史 DSO 身份：

```bash
PYTHONPATH=/source/src:/source/fault39/src:/source python3 -u \
  /source/experiments/worker_core_v11_20261118/output39_refusal_probe.py \
  --model /model --library /native/libdcat_postwrite.so --sha256 NEW_DSO_SHA256 \
  --build-fingerprint NEW_BUILD_FINGERPRINT --out /evidence/probe \
  --deadline-seconds 1500
```

[`prepare_output39_refusal.py`](prepare_output39_refusal.py) 可生成通用 queue 的**一个** case，复用现有只读 source/native/model mount、单卡门禁和原 image/CANN 环境，不新增重复 collector mount：

```bash
PYTHONPATH=/source python3 -m experiments.worker_core_v11_20261118.prepare_output39_refusal \
  --source /ACTUAL_NEW_SOURCE --native /ACTUAL_NEW_NATIVE --model /ACTUAL_UNCHANGED_MODEL \
  --sha256 NEW_DSO_SHA256 --build-fingerprint NEW_BUILD_FINGERPRINT \
  --card 2 --out /NEW_ATTEMPT/refusal-case.json
```

生成器仅准备文件、不执行 NPU。root 调度者必须照旧进行资源锁/门禁，不能直接绕过队列。故障窗口软超时1500秒，queue 外层与 `resource_budget.timeout_seconds` 均保持既有硬预算1800秒；预算不足则如实失败，不能扩到2400秒。Linux 主线程 timer 不覆盖已有 timer，恢复阶段取消自有 timer。Python软超时不能保证打断阻塞的 Native 调用，外层硬截止仍由原controller负责，超时且无法确认恢复时保留 owner/资源未知结论。协议请求和观测条数另有独立计数上限。

queue manifest 的 `phase_budget` 列出当前原公共 plan 的全部42个 group 请求以及截至首个 `c3_fault` 的39个请求前缀、最多2个原恢复请求、安装前与注销后各1个普通健康请求，从而推导总上限43。安装时 helper 读取**真实 provider.plan**重新计算并核对，原计划仍包含第二轮C3及其OFF/health；诊断在原拒绝处停止，不重新执行一个缩减的替代协议。

runner 始终以 exit 2 保留这次累计请求未完成的事实。`POSTWRITE_APPLY_4` 是待诊断的原失败，不将其改为成功退出或 C3 通过；独立 reader 的 pattern 结论另列。若 trace 丢失、实际线程不同、原 source pin 不匹配或 DSO 未含新 export，停止并保留错误，不放宽检查。**当前 runner 已做 CPU 合同，尚未编译真实 SDK DSO 或启动124诊断。**

## 独立判断标准

固定输入、权重、软件/构建、model/Worker 生命周期和真实请求映射需由原完整请求证据关联。该 sidecar 不是它们的替代品。

1. 健康第1层 span 有限；单前序 F 的第1层 fresh span 出现非有限；累计 F 第1层 **同一 lease.original** 出现非有限并真实 return 4。BF16分类/首位置/全 span 摘要应内部一致。
2. 单F与累计 F 的第1层完整 span SHA相同；两个臂第0层 mutation 之前的完整 original SHA、模式/索引和实际效果相同。若只统计数量相同而完整 SHA不同，不给严格同快照结论。
3. 原全量 Native/coverage/RPC 原件独立证明单F臂恰一个实际写入，累计臂拒绝前也恰此前序写入；不凭这5条窄观测推断整个请求没有其他干预。实际 producer/consumer、同一存储值读取及固定分母必须可关联。
4. 失败 apply 不伪造下游 consumer。正常 OFF、health 和资源释放单独核验；恢复成功不能追认累计 C3 成功。

满足这些条件时，允许的结论是：“在本次固定输入及新构建下，单个前序 F 足以产生与累计拒绝租约相同的下游原快照，拒绝与该传播相符。”仍不能据此宣称历史私有 OpCache 缺陷、每个中间传播算子、所有模型/输入、原 source84 的完整原数组已经被重建。任一对齐或来源缺失则结论为**因果未闭合**，不改阈值、不换位置继续找结果。

[`review39_refusal_diagnostic.py`](review39_refusal_diagnostic.py) 是独立 CPU metadata reader，固定检查五条观测、实际请求/输入、位模式分类、完整 SHA、原 return4 与 consumer 边界。它只报告 counterfactual pattern 一致性，并始终保持 `c2/c3/e5/causal_qualification_granted=false`。CLI 必须 pin 输入文件 SHA：

```bash
PYTHONPATH=/source python -m experiments.worker_core_v11_20261118.review39_refusal_diagnostic \
  --report /evidence/refusal-diagnostic.json --sha256 ACTUAL_SHA256 \
  --out /evidence/refusal-diagnostic-review-new.json
```

## CPU 验证与当前范围

```bash
PYTHONPATH=src:fault39/src:.:tests:tests/worker_extension \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -B -m unittest \
  test_output39_refusal_diagnostic test_output39_original_native_source -q
```

实验 runner 及 helper 的实际原公共 driver 合同：`python -B -m unittest test_output39_refusal_probe -q`。覆盖6前置后安装、恰一次累计请求、原拒绝/恢复、未复现时明确停止、写文件失败不阻止恢复、deadline/第二次累计请求拒绝、单case准备。

CPU 合同使用当前真实 public Host/Runtime、实际 Torch 模型调用与明确标注的 CPU ABI double，覆盖完整C2、首累计拒绝、原 OFF/健康/清理，以及错误源码、外来 trace、错误分母、metadata漂移、旧export缺失等拒绝路径。原件 reader 的负例包括缺控制、不同完整 SHA、错 request/input、虚构 consumer、fresh 伪装 original、apply4 改0、null计数及错误位模式。

独立 [`test_original_snapshot.cpp`](../../fault_injection/native_adapters/postwrite/test_original_snapshot.cpp) 直接包含实际 `postwrite.cpp`，在显式 CPU ACL stubs 下核对同租约原快照、无额外 ACL 调用、跨线程/外来owner/错缓冲区/别名拒绝，原 return4/5 以及写入/恢复不变。Python 测试在有宿主 C++ 编译器时自动编译运行；当前本地 Windows 无编译器，该项 skip，**不是 Native 编译通过或 NPU 通过**。root 的生产无设备 CPU 镜像应先运行该项，再考虑真实 SDK 构建和一次124诊断。

最小额外导出闭包为本目录4个 Python 文件 `output39_refusal_diagnostic.py`、`review39_refusal_diagnostic.py`、`output39_refusal_probe.py`、`prepare_output39_refusal.py`，以及 Native 的 `postwrite.cpp/.h`。其余依赖是现有原公共 driver、完整 output39 provider 栈和标准库；没有新 pip 包。CPU运行另带 `test_original_snapshot.cpp` 与3个新增 Python tests；测试复用原 fixture，其依赖已在现有 tests 闭包内。

所有新诊断、原始张量与 RPC 原件仅留验收机；公开交付仅实现、测试、摘要和本文。E5 与本诊断完全分开。
