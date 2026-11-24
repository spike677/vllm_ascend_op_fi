# FIA 输入只读 journal：CPU 候选与接线边界

这是一项独立、默认关闭的观察候选。它没有接入公共 registry、census、provider 或故障开关；不创建 Worker，不包裹全局 `execute_model`，不改 Q/K/V、mask、输出、tiling 或调用顺序。当前交付是实现和 CPU 合同，**没有新的 NPU 结果、正常桥接资格、C1/C2/C3 或 E5**。

目标是保留真实 public API 窗口中的 FIA V3 GWS 参数、容器创建和销毁、execute 配对。原始 V view 的 stride `[4096,128,1]` 和 offset `3072` 必须照实记录，不能经 `contiguous()` 后替代原对象。journal 能比较 public 存储几何与 GWS 参数/creator 快照；不能据此证明 executor 内部 prepared tensor，也不能补齐 Slice/copy 输入血缘。

## 已有 ABI 依据

生产 TU 直接包含 `aclnnop/aclnn_fused_infer_attention_score_v3.h`，使用 `decltype` 得到真实函数类型，包装器声明由同一 SDK 编译检查。当前候选只接受实取 CANN 8.5.1、既有固定镜像和 owner 文件指纹，不泛化声明其他版本。

- V3 头 SHA256：`bf73aea4146a9b12f91eda466927c23c5d43e56f78aa5039bd9644732f521550`。
- 真实 SDK 材料包 SHA256：`aed0d9a6e28bbc22d12e2ca40a777034d05d62851fc1251f523389dc7377da44`。材料来自隔离无设备构建环境；`g++ -MM` 得到的七头闭包逐字节固定。构建器另固定实际 `acl_rt.h`。
- 历史 source75 头指纹摘要 SHA256：`551d31cf0b4f4cd9ac59befcb869e256bf793ffc7d5c2e589cd1e9e1f6c329ee`。其中 `inputs.json` 的主/level2 V3 头同为上述 SHA。这只证明头字节一致，不能替代新进程的原函数 owner 绑定。
- SDK 头和私有原件不入 Git。新增构建回执必须区分源码、SDK、镜像、observer、实际原函数 owner 与运行期身份。

44 个 GWS 参数分为 24 个 `aclTensor*`、5 个容器、10 个有符号整数、scale、layout、bool 和两个输出指针。5 个容器是 K/V `aclTensorList`，以及 query/KV/shared-prefix 长度 `aclIntArray`。只有已提供的公开 size getter；没有假造 list 成员 getter，不扫描 opaque 内存。

## 实现与失败规则

`fault_injection/native_adapters/fia_input_journal.cpp` 包含字节未改的四容器 journal 实现，复用 SDK descriptor getter、OwnedArray 和实时 registration。它不启用 GMM2 extension 或 writer26 callback。

1. 独立 DSO 预加载，先进行 NOLOAD 原函数解析、文件/可执行段 owner 检查，再允许创建 LLM。解析失败应在启动门禁拒绝；**不能称所有启动失败仍会透传**。
2. 同一进程/线程的显式 owner 开始后，每个窗口沿用已有 launch observer 的真实 32 位十六进制 `call_id`，记录 Worker/model/module 对象、forward、operator/boundary sequence 和逻辑设备。它不是新建的同名 token。
3. 新增的 FIA GWS/execute 包装器对原函数恰好调用一次，参数、返回码和原异常不替换。观察器 owner、getter、预算、线程等失败记录下来，仍执行原函数，不能把这些调用写成完整观察。这个异常透明结论只针对新增两函数；复用的旧四容器 TU 原有 creator/destroy 异常 catch 会转为 `nullptr`/`161001` 并记错，本补丁没有更改或掩盖它，不能泛称整个 DSO 对 C++ 异常透明。
4. K/V 成员来自实际 creator entry 的指针数组，带完整 descriptor 元信息；GWS 时用当前 live registration、size getter、同窗口/线程和 destroy 寿命连接。不会在 execute 时重新解引用已经可能销毁的 descriptor。容器地址重用必须有新 registration。
5. 成功 GWS 产生的 executor 保留到同窗口/线程/设备、足够 workspace 的一次成功 execute。重复 executor、无配对、执行失败或未知状态不丢弃。GWS 失败/抛异常时不读 SDK 未保证初始化的输出参数，记录 `failed_prepare_outputs_unknown` 和未知 pending；保留 owner 到独立进程退出，不能假称已释放。
6. stop/release 必须无活动调用、scope、pending/untracked executor、live container。读取原快照后另存 release 回执，核对同 owner 和计数、getter 数组平衡、两个 journal 缓冲区归零。全局 wrapper 和持有的 DSO 保留至进程退出，未宣称全局卸载。
7. Python 控制器逐阶段保存真实 return code、操作前后两组件状态及首个错误。base 已安装而 FIA begin 失败时，只在实时查询证明 FIA 从未安装、无 pending/缓冲/token，且 base 无 scope/active/live 时才可单独撤 base。查询不可用不能当零状态；FIA 已成功释放而 base 暂时失败时，重试核对本次原生 release 回执及同 owner 的 retired/零资源状态，只继续 base。失败原件不覆盖，未知资源不 reset，实例不重新安装。
8. `max_windows` 独立于原生调用预算，1–256、默认 Python 64；空窗口也占用一个 token。Native 明确报告 accepted windows、retained token 数及 token 字符数，release 后后两者必须归零。字符数不是 allocator 总字节数；额外 set 节点由窗口数硬限制。Python 同限，每个 scope evidence 最多 64 KiB；生命周期控制和每窗口 leave 最多各 64 次，超限保留 owner，不再追加无界日志。

C++ CPU fixture 直接包含这份生产 TU，替换的只是原数值函数、getter 和 owner 函数指针；不通过生产宏绕开 journal。26 个独立进程用例包含默认未启用、正常非连续视图、getter 返回失败/赋值后抛异常、缺失成员、重用地址、外线程、错误 owner/caller/token、重复 executor、失败/未初始化 GWS 输出、execute 失败/重试/抛异常、workspace 和预算不足、活动中清理拒绝，以及无效/空窗口的硬上限和 token 释放归零。

## Root 可执行的无设备构建

只在 root 已安排的无设备 SDK 容器中运行。脚本自身不启动容器、不 SSH、不加载构建出的 DSO，不执行 NPU 算子。observer 为当前已核验的正常 launch observer，其路径和 SHA 必须由本轮 manifest 提供，不使用占位构建当运行 owner。

```sh
python /source/experiments/worker_core_v11_20261118/fia_input_journal_build.py \
  --source /source --sdk /usr/local/Ascend/cann-8.5.1 \
  --observer /actual/read-only/libattention_launch_observer.so \
  --observer-sha256 ACTUAL_OBSERVER_SHA256 --out /output/fia-input-journal-fresh
```

导出闭包由 `fia_input_journal_build.SOURCES` 明确列出：新增 native 三件、Python 配置/独立 reader/build；原容器 native 三件、`quant_container_native.py`、`metadata_arrays.py`；`fault39/native/{runtime_identity,metadata_array_owner,metadata_array_state}.h`；`version_profile.py` 及既有容器构建工具和其工具依赖。root 冻结的完整 source archive 仍是最终源码身份。额外测试为 `tests/test_fia_input_journal.py`。

构建必须成功运行 26 个 production-TU CPU 用例、保留每个日志和 executable/DSO/SHA，再生成 `build.json`。本地 Windows 已运行 37 个 Python 合同，通过，含实际配置器控制流上的 return-code doubles 和分阶段失败/重试；当前交付时尚未在本机编译 C++，不把声明的 C++ 用例数算作已通过。Limits/State 已新增窗口字段，旧候选二进制不可复用，须 fresh SDK 编译与 layout/NOLOAD 校验。SDK 编译及实际调用均待 root 独立执行。

## 未来接线位置与原件

本补丁不修改接线。将来只能接在已有确切 `attention28_stock_census` public 原调用边界，并在 launch scope 已创建之后、原 API 调用之前建立 journal 窗口。顺序为：

```text
preloaded observer + journal
→ journal.preflight()，成功后才构造 LLM
→ 绑定同一实际 Worker/model 对象与 process-start identity
→ journal.begin()
→ 已有 launch scope 的实际 call_id
→ journal.enter(actual identities, actual input metadata)
→ 原 public FIA 恰好一次
→ 原调用完成/失败的原始记录与必要设备同步
→ journal.leave(original_calls, original_error)
→ 无 pending 时 stop → 保存 snapshot 原件 → release → 保存 release 原件
```

`public_tensor_metadata()` 只读真实 tensor 的对象、StorageImpl 身份、storage/data 地址、dtype/device、shape/stride/offset/bytes，没有复制/重排或设备读。调用方必须把同一对象在原调用前后保活，保存原输入引用及 RPC/模型自然调用链；配置中传入一个对象 ID 本身不是调用证明。`evidence.actual_input_addresses` 分别使用 `query/key/value/atten_mask` 键。缺失、几何不等或 dtype 不支持只报告未 join，不补成真。当前 helper 支持 reader 明确比较的 FP16/int8、ND 元信息；其他类型需实际 SDK 类型合同后再适配。

异常时保留原异常、失败快照、退出和资源状态，不覆盖为成功；若 pending 导致不能 leave/stop/release，不强制 reset，也不复用此 Worker。无需为观察候选增加 F。不能由本地假对象或手写输入几何生成 NPU 原件。

独立纯 stdlib reader 不导入配置器或 `assess_observation`，不使用命中/期望输出。它核原件 SHA、结构/顺序、44 typed slots、容器原始 pair/寿命、输入几何、真实 execute 和 release。每个 public 窗口都必须有对应 GWS，缺事件的窗口不能被其他窗口的成功覆盖。K/V creator 成员数不为 1 时仍在固定四个 public role 中保留，明确未 join，不让分母缩小；原生记录完整但 public 几何不全时使用 `COMPLETE_NATIVE_JOURNAL_PUBLIC_JOIN_INCOMPLETE`，不能称 public 参数关联完成。调用示例：

```sh
python /source/experiments/worker_core_v11_20261118/fia_input_journal_review.py \
  --observation /run/fia-input-observation.json --observation-sha256 ACTUAL_SHA256 \
  --released /run/fia-input-released.json --released-sha256 ACTUAL_RELEASE_SHA256 \
  --build-root /actual/read-only/build --build-sha256 ACTUAL_BUILD_SHA256 \
  --out /review/new-summary.json
```

失败无法 release 时可不提供两项 release 参数，reader 保留部分失败和缺失项；不能输出完整清理结论。所有结果始终 `C1/C2/C3/E5/qualification_granted=false`。即使 public/GWS 存储相同，**Slice ABI、内部 prepared tensor 和 copy lineage 仍为未知/false**；没有数值统计比较，也不授正常模型或故障验收。
