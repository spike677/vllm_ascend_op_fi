# 自动模型适配与统一故障入口

[文档索引](README.md) · [生产接线规划](#生产接线规划)

后续生命周期与宿主接线规范见[其余15个机制合入说明](remaining_faults_integration_20261116/README.md)。
清理失败会保留隔离与待恢复状态，不能把退出异常解释为已经恢复成功。

本轮将本地故障资产纳入一个能力目录，并增加按模型对象自动定位的接入层。无需用户填写 `model.layers.0.self_attn`、层号、头数或Qwen专用注册代码。接口统一不意味着全部旧故障已经迁移，具体状态可运行 `python -m fault_injection` 查看。

后续其他机制的实现、自动选择方式及逐项验收边界见[2026-11-16扩展说明](unified_mechanisms_20261116.md)。公开入口统一为`attach(model, fault, arm, unsupported, evidence_dir)`，部署侧另可提供`runtime`，不向模型用户索要层名。

在已经匹配 torch/torch-npu/Transformers 的推理环境中执行 `pip install --no-deps -e .` 安装本地接口；不会自行替换设备依赖。只读清单和策略框架本身不导入torch或初始化NPU。

```python
from fault_injection import inspect_model, attach

print(inspect_model(model))  # 只读；模型可为Qwen、GLM、DeepSeek或其他结构
with attach(model, "28.fia.denominator_replay", arm="fault") as session:
    result = model(**inputs)  # 使用原模型调用，不另造forward
    session.assert_coverage()
print(session.report())
```

`baseline` 与 `fault` 使用相同的模型接口。`assert_coverage()`确认已发现的候选模块实际经过路由；不证明kernel命中、历史根因或msprobe首差。默认遇到不支持的调用立即报错；显式 `unsupported="passthrough"` 仅允许能力预检拒绝的调用返回原后端，报告中会记录，不能算全覆盖。进入后端执行后发生的异常不会偷偷重跑原算子。

## 模型发现与掩码语义

1. 扫描 `named_modules()`，分析实际 `forward` 的共享注册表调用，核对真实注册表对象，而非模型名称白名单或固定属性路径。
2. 使用公开 `transformers.AttentionInterface.register`；同时为临时后端名称登记原后端的 `AttentionMaskInterface` mask factory。只注册Attention名称可能导致上层丢弃mask，不能接受。
3. 回调从实际Q/K/V读取请求数、Q头/KV头、序列长度、QK维度及V维度。DeepSeek的QK与V维度分开描述；不通过pad/crop强行适配。
4. 只控制发现的模块对象。共享config的未知模块、绕过注册表的remote-code实现、未知mask后端默认拒绝并给出原因。源码检查是候选发现，真实回调才是覆盖证据。
5. 退出、模型异常或安装失败均恢复原config与自身注册项。嵌套/并行session拒绝，避免进程共享注册表冲突。不要在同进程中同时运行共享这些config的其他模型。

目前路由支持从原 `eager`/`sdpa` 后端接入；其他后端需核验掩码语义。运行环境需要支持本文协议的 Transformers，测试版本固定为5.5.4。公共接口可能跨版本变化，不能从单版本测试推导所有版本兼容。

## 原生执行策略与FIA边界

`fault_injection/strategy.py`复用已有`native_execution_strategy.StrategySession`，原框架文件逐字节复制，来源与SHA在`native_execution_strategy/IMPORT_MANIFEST.json`。FA桥接读取真实调用参数和已验证请求映射，保留合法候选、证书、原生raw和launch门槛。没有另造一套可绕过旧检查的策略接口。它是平台提供者接口，尚未自动接入所有模型中的原生FA调用。

FIA分母重复累计由`fault_injection/fia.py`接入。Python层按真实模块位置自动路由，kernel仍只接受B1/B8、Q16/KV8、2048-token prefill、QK/V维度128、FP16、causal、固定scale及匹配版本/构建。故障位于真实NPU kernel，Python不修改返回数值。新模型即使能注册，几何不匹配也会明确拒绝。模型品牌不参与kernel选择。

需要通过既有不可变vendor启动器启动新进程（不能在已初始化进程换库）。部署配置由运行环境提供：

- `DEMONCAT_FIA_VENDOR_MANIFEST`：已经验收的分母重复累计vendor清单路径，固定SHA在代码中；依赖完整原vendor，不能只下载证据分支中的源码。
- `ASCEND_CUSTOM_OPP_PATH`和动态库路径：由已有验证启动器设置；`ACLNN_CACHE_LIMIT=0`。
- `DEMONCAT_EVIDENCE_DIR`：`/evidence`下专用目录，自动产生逐调用scope与host日志。

这些是构建与证据目录设置，不是让模型用户定位注入层。新会话仅报告Python路由及host eligibility，`native_launch_verified`保留false；实际launch和B/F/msprobe验收仍使用独立原生观察工具，不能拿host命中日志替代。磁盘vendor校验也不等于实际加载身份验证。

现有native策略是进程作用域，必须使用专用推理进程，禁止其他线程/服务在此进程并行调用FIA；Python门禁拒绝已知活动线程与嵌套会话，但不能替代C++调用方隔离。host回执严格核对唯一scope、owner、mode、M、key和serializer。部署到并发服务之前仍需独立的native调度作用域适配，不能直接套用。

旧Qwen3/124的NPU结论属于旧冻结入口。本轮新的通用路由没有获得GLM、DeepSeek、Llama的NPU验收。图模式、分页decode、滑窗、padding、训练dropout等不属于当前FIA故障kernel支持范围。固定策略下也不保证每份输入产生差异。

## 完整故障整理

统一能力目录将注册与执行分开：

| 状态 | 含义 |
| --- | --- |
| `automatic_attention` | 通用模型路由已实现；实际调用还要过kernel门禁 |
| `automatic_operator` | 在真实模型作用域观察API/存储事实；目前为15公开as_strided |
| `automatic_module` | 按标准模块及真实参数选择；目前为17/18新本地重建实现 |
| `automatic_quantization_requires_runtime` | 实际量化method自动选择；仍需部署后端及调度provider |
| `native_provider_bridge` | 自动候选与原生callback桥接完成；具体版本provider待适配 |
| `strategy_bridge` | 旧原生执行策略已桥接；需要平台实际调用/launch观察接入 |
| `legacy_source` | 有旧实现及限定验收；尚不能由新入口自动启动 |
| `evidence_only` | 当前仅有历史证据，不能虚构可执行插件 |
| `not_implemented` | 方案存在，但没有可运行且验收的适配器 |

15的旧27误称、28合法切分与新的分母重复累计、33分页decode分别登记。20/26/33/39已增加统一桥接，仍需实际ABI、slot、存储、分页与生命周期证明；发现一个Linear或Attention模块不能代替这些事实。完整条目和源位置见[清单](unified_fault_inventory.md)。

## 验证与统计

结构回归、mask保留、拒绝策略、恢复、覆盖计数及FA桥接测试：

```bash
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s native_execution_strategy/tests
```

真实模型测试使用本地随机初始化的小模型，只验证接口语义，不下载权重，不占NPU，不代表模型质量或原生故障泛化。首次整理结果：**35项新接口测试＋69项原策略回归通过，无跳过**。见[旧验证回执](unified_validation_20261116/receipt.json)。后续机制扩展结果单独保存在[新验证回执](unified_mechanisms_validation_20261116/receipt.json)，不覆盖旧日志。

| 真实CPU模型代码 | 测试层数 | 验证内容 |
| --- | ---: | --- |
| Qwen3 | 3 | eager与SDPA两条路径；GQA、padding、prefill、cache decode及恢复 |
| GLM | 2 | eager；不同head_dim的实际几何、padding/cache与恢复 |
| DeepSeek-V3 | 2 | eager；QK维12、V维8分别传递、cache与恢复 |
| Llama | 2 | eager；GQA、padding/cache与恢复 |

测试环境为torch2.10.0+cpu、Transformers5.5.4、NumPy2.2.6。Windows最初的NumPy2.5.5扩展无法加载，固定依赖后重新完整执行；没有改动系统安全策略。可用 `python -B tests/run_unified_validation.py --output NEW_DIRECTORY` 生成新的带源码SHA回执。

msprobe仍在外部按同配置采集B/F的原始 `statistics/mix`，保留dump/stack/construct、配置和collector身份。新接口动态解析公开NPU API，避免缓存msprobe安装前的函数。覆盖报告与注入日志不进入独立首差的输入。历史统计原件及固定容差比较器均未修改。

公共注册与mask接口依据：[Transformers Attention backends](https://huggingface.co/docs/transformers/attention_interface)。GLM和DeepSeek共享接口的存在不等于当前故障kernel已支持其全部调用。

## 生产接线规划

本分支已实现离线宿主准备。Fault20 可选 scoped Provider 接入已有 `register_runtime`、`ModelFaultBackend` 和 `NativeSession`，两次发现与无损回放通过后只消费一次机会。Fault39 的 `DocumentContractAdapter(..., qualified=True)` 在既有 binding 上使用 `QualifiedFault39Backend`。真实宿主需要提供 Worker 独占作用域、实际模型与 Worker 身份、scheduler 请求和完整 readback。两条适配均未获得黄区 CLI 验收。[分支证据](../experiments/production_wiring_20261116/README.md)

### 第一条样板 · Fault20 MatmulWeightNz

实施依赖黄区实际 `src/demoncat` 的目标提交、真实 CLI/schema、现有 PrimitiveSpec、RuntimeBinding、Controller 和 Worker 扩展入口。先确认实际模型驻留在哪个 Worker、RPC 在哪条执行线程进入、关闭能否等待原生在途调用。源码和部署环境齐备后复用现有调用链。

```text
现有 CLI → 现有 Catalog / Binding → 现有 Controller
  → Worker RPC → 所属 Worker 的 ModelFaultBackend / NativeSession
  → Buffer20Runtime → typed GWS / live Run → OFF 与恢复回执
```

1. **锁定部署身份。** 使用审计过的 `9171f3b`、`build20-a1` 和 CANN 8.5.1。把 `provider-source-a3` 的 Python、`provider-source-a1` 的 bridge/Run、实际镜像、aarch64/metadata ABI17、ABI 预加载及主库/split/real DSO 的实际加载身份串到同一实例。现有源审查覆盖核心 Python 和 native 文件，仍需实测环境确认实际映射与构建脚本及 metadata proxy 的来源。
2. **接入现有宿主。** 注册 `model_fault.20.buffer_substitute` 对应的 Primitive/Binding。实际 Worker 用 `prepare → register disabled → enable`。CLI 只提供机制和已经被实际 binding 消费的作用域，Controller 按真实 rank/device 路由，Worker 从已加载模型确定实际 Linear、forward、executor、formal self/out slot 和 cache 生命周期。所有 `query` 回执带 run、Worker、owner、generation、构建、选中范围和实际原生结果。
3. **执行冻结样板。** 从已验证 Qwen3-0.6B BF16/NZ29、B2 单 token、`use_cache=False` 开始，在单卡专用同步模型 Worker 上走一次有界 workload。v17 bridge 会在逐调用执行中短暂开启进程 SHM arm，故该样板使用隔离 IPC 和同线程、无竞争原生调用。旧 `max_hits=1` 约束单个 arm；首个生产样板的调用范围、实际数量和结论按该请求明确记账。
4. **完成 B/F/OFF。** 用相同模型权重、请求和初始缓存比健康参考、无损 D2D、F 与同进程 OFF。实际 GWS/Run 记录应证明 formal self 到当前 slot、替代 buffer 和原 out、Run 恰好一次及资源释放。发布回执保护了用户 token 输入并核对完整 logits，新验收另增加选中 self/mat2 原 storage 字节核对。
5. **验收真实退出。** 由 CLI 驱动 query、disable、recovery、clean。在途归零后复跑原始输入，核对输出精确恢复、lease 释放、owner 解绑和其他故障未干扰。用错误 owner、旧 generation、异线程、超时、unsupported、同形状不同 cache 和恢复失败做拒绝回归。超时或部分 Worker 清理失败时保留 owner/lease 的隔离和查询。

第二个模型 SmolLM2 复用同一 binding，保留独立模型和证据配置，验证迁移。生产样板要求严格唯一、一次命中。v17 是已实测对照，新的 scoped WeightNz 构建须先通过单点实机测试。其尾数参数、Run 拦截和 IPC 边界与发布试验各自留档。QMM 必须另验 `aclnnQuantMatmulWeightNz` 的 x1/量化参数/out 等实际槽位、读回和 B/F/OFF。[发布部署说明](../fault_injection/deployments/README.md) · [构建与回执](../experiments/provider_adaptation_20261116/validation_summary.json)

### 后续接线 · Fault39 MatMul

- 使用同一 CLI、Catalog、RuntimeBinding、Controller 和 Worker 入口。使统一 `ModelFaultBackend` 的该机制 binding 调用现有 `NativeBackend`、`WorkerEndpoint`、`Fault39WorkerExtension.fault39_rpc`。保留 `fault39.native.matmul` 的真实 controller、调度上下文、设备身份验证及 run-slot 证明。发布版 `FaultConfig` 拒绝 `39.*`，实施时只为 `39.matmul_range` 接入已审核的类型与路由。
- 原生 ABI v1（Config 368 字节）及旧包继续可用。配置 owner、generation、真实 worker/rank、device、request 和 hit budget 都须进入实际 native 门禁，失败时逐 rank 回滚。先 TP1、一个合格稳定目标，最多一次尝试。严格 `unique` 遇到多个候选返回歧义。发布真模型部分用例显式选了 `max_k_min_m_unique`，因此要用对应策略重新标注。
- 现有独立运行脚本在 enable 外核验 lossless。生产接线须消费同模型、Worker、profile、slot 和缓存代次的 self-only、mat2-only、dual 设备复制与读回记录，健康/无损/OFF 输出一致后才允许 F。随后对 TP2 按实际 rank 重放资格和部分失败恢复。保留实际 MatMul 有限输入替换与结果观测。
- 本地 `QualifiedNativeSession` 的同步卡位、D2D 和一次尝试原型可供接线参考。它需完成新的 CANN 构建与 124 实测。独立包的四版本算子和两架构报告继续按各自实现/版本归属。[独立合同](../fault39/src/demoncat_fault39/contracts.py) · [黄区映射](../fault39/docs/yellow_integration.md) · [当前 CANN 版本证据](../fault39/docs/cann_version_adaptation.md)

### 最小代码触点

| 位置 | 实施内容 |
| --- | --- |
| 黄区现有 CLI、Catalog、Binding | 按真实 API 登记两条 approved binding，校验已消费的请求参数，query 返回实际 Worker、owner、generation。 |
| 黄区现有 Controller / Worker 扩展 | 接上 scoped backend，提供同步原生回调环境、真实 rank/device 路由、disable/drain/recovery 和失败隔离。实际路径待源码确定。 |
| 蓝区 `fault_injection/integration.py` / `lifecycle.py` | 20 先复用已有 `model_fault.20.buffer_substitute` 合同与生命周期。39 接线时增加单机制映射及资格回执接入。 |
| 现有 Provider / 独立 Fault39 包 | 复用已审核冻结实现做初次生产样板。新隔离式 native 回调按独立构建验收。 |
| 现有 docs 和 experiments | 当前审计写入故障总账、本文及覆盖标准。未来实测在现有实验目录登记 run/build/compare/recovery 证据。 |

### 测试矩阵及资源

| 试验 | 通过标准与资源 |
| --- | --- |
| 本地只读合同 | 17 个变体与 15+1 合同及 1 个缺项对上，实机证书独立归属。已有蓝区合同和覆盖单测供回归。 |
| 真实宿主 CPU/RPC 流程 | 真实 CLI register/prepare/enable/query/disable/clean，未知配置拒绝，独占 owner、generation 与部分失败恢复。需实际宿主和可运行 Worker 测试环境。 |
| 20 生产样板 | CANN 8.5.1/aarch64 匹配 bundle、专用同步 Worker、空闲 124 单卡窗口、Qwen3-0.6B 和第二模型权重。真实 slot/Run、B/F/OFF、租约与原始统计独立复核。 |
| 39 生产样板 | 同一宿主，先独占 TP1 一卡与实际 profile。self/mat2/dual 后命中一次。TP2 用同一资源锁和两个空闲卡，含错 rank 与部分 Worker 清理回归。 |
| Provider 定向补测 | 20 QMM 新回调。26 QMM rank1 step0 原始 NaN 与 FIA 异常 consumer；量化请求重排、33 原生 IFA 和 page/cache OFF。匹配镜像与独立采集。 |
| 多目标资格 | 按机制单独验证 `all_eligible`、`every_match`、资源预算、多 rank 和整组恢复。先在同 Worker 有界串行执行。 |

生产实施当前缺实际宿主源码、目标版本和可运行环境。实机使用 124 现有资源锁及逐例空闲检查，保留外部任务。新私有统计留在验证机，GitHub 记录相对路径、摘要、hash 与比较器。[发布基线审计](unified_fault_inventory.md#9171f3b-全局审计) · [覆盖与恢复标准](fault_coverage_acceptance_standard.md)
