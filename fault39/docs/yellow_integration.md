# Fault39 黄区接线与独立验收

本包交付可复用的 Fault39 核心、蓝区 Worker 接入和验证工具。蓝区没有当前
内源 `src/demoncat`，因此状态仍为 **`INNER_FRAMEWORK_INTEGRATION_PENDING`**。
本文是接线契约，不是对黄区现有类名、方法签名或完成状态的声明。实际结果以
[主 README](../README.md) 和对应实测证据为准。

## 1. 对照实际内源实现完成接线

| 黄区职责 | 本包已有入口 | 接线要求 |
|---|---|---|
| Primitive catalog | [schema.py](../src/demoncat_fault39/schema.py) 的 `PRIMITIVE`、`Fault39Config`；[contracts.py](../src/demoncat_fault39/contracts.py) 的 `integration_manifest()` | 仅登记 `FLOAT_MATMUL_CANCELLATION` 一个 primitive，fault_id=39；五种 pattern 是参数，不能拆成五条不同机制 |
| Binding registry | `DocumentContractAdapter.install()`、`HostContract` | 将 `fault39.native.matmul` 绑定到 `NativeBackend`；实际宿主实现需自行映射内源 registry API，不能假定示例 protocol 就是内源 API |
| Controller | [backend.py](../src/demoncat_fault39/backend.py) 的 `NativeBackend`、`WorkerEndpoint` | 管理 owner、真实 Worker 映射、generation、发现结果、启停和失败回滚；不得复用其他 fault 的全局清理 |
| Worker | [vllm_worker.py](../src/demoncat_fault39/vllm_worker.py) 的 `Fault39WorkerExtension.fault39_rpc()`、`prepare_worker()`、`install_runner_scope()` | 组合进既有 Worker Extension，避免覆盖其他扩展；核实真实 `rank`、逻辑 device 和执行线程 |
| Native provider | [provider.py](../src/demoncat_fault39/provider.py) 的 `CtypesProvider.attach_preloaded()`、`configure()`、`context()` | 连接已经预加载的符号，消费筛选条件，围绕真实调度任务设置并清除请求上下文 |
| 检查与报告 | [inspector.py](../src/demoncat_fault39/inspector.py) 的 `Fault39OutputInspector.inspect()`；[compare_msprobe.py](../tools/compare_msprobe.py) | Native 证据与 msprobe 首差分别判断，之后再核验真实请求关联 |

`HostContract` 中的 `catalog_register_primitive()`、`binding_register()` 等名称
属于版本化文档协议 `demoncat_fault39.host_contract.v1`。黄区需实现这一适配层，
或在实际框架中实现等价契约并测试。不能把 mock host 的安装成功当成内源接线成功。
卸载 catalog/binding 之前，应先对所有已注册 backend 完成 scoped clean/unregister；
`DocumentContractAdapter.uninstall()` 本身不会替 controller 排空 Worker。

## 2. 生命周期必须消费真实状态

1. `prepare()` 检查已加载 Native 符号和 Worker 身份。加载成功不等于拦截成功。
2. `register(Fault39Config)` 仅登记配置，保持关闭。**Native `configure` 会开启对应
   discover/observer/lossless/inject 模式**，不能把它直接当成“默认关闭”的注册动作。
3. `discover()` 后执行真实代表性工作负载，调用 `observe_inventory()`，重复工作负载后
   再调用一次；相邻两次控制查询不能证明目标稳定。`end_discovery()` 需要同步关闭。
4. 只在目标和无损门禁通过后 `enable()`。使用每个实际 Worker 返回的 generation，
   不从 controller 的本地计数推算。
5. `disable()` 必须关闭新命中并等待在途 Run 和 buffer lease 清零。仅翻转 enabled
   flag 不能回报成功。超时保留 `DISABLE_PENDING`；同步、释放失败保留 `RECOVERY_FAILED`。
6. `clean()` / `unregister()` 仅处理当前 Fault39 owner。临时输入不会写回原模型权重，
   不调用全局 `reload_weights()`，也不清空其他 fault。失败租约需要原线程、device、
   ACL context 中的显式恢复，不能由任意管理线程直接释放。

部分 Worker enable 失败时，保留 [NativeBackend._arm()](../src/demoncat_fault39/backend.py)
的回滚和未知应答处理：丢失 configure 回执后仍需查询真实 owner/generation；无法确认的
Worker 继续 pending，不能伪报整体关闭。

## 3. CLI/API 筛选条件不能只是状态标签

黄区实际 CLI/config 应显式映射到 `Fault39Config`，并拒绝未消费字段。下表描述
数据流，不声称已经提供对应内源 CLI 命令。

| 外部条件 | 必须消费的位置 |
|---|---|
| rank 列表 | Controller 选取 `WorkerEndpoint`；RPC 检查 `dispatch_rank`；Worker 真实 rank；Native immutable worker-rank binding 与 Run 时筛选 |
| physical card 列表 | 启动器记录物理卡到 Worker 逻辑 device 的真实映射；再传入 `devices` / WorkerEndpoint。不能把物理卡号直接作为容器内逻辑序号 |
| pattern / amplitude_policy | 进入 Native Config；所有同轮 pattern 使用同一幅值规则与指数，记录实际结果 |
| max_hits / max_bytes | Native 命中上限与实际存储合法性检查；不能只显示在 query 中 |
| 请求作用域 | 来自真实 scheduler 的 request IDs；不得从用户配置或时间戳编造 |
| target | 只支持 `auto`；不要求用户填写模块层名。未实现的 op/callsite 字段不得静默透传 |

当前 Worker 对 `num_scheduled_tokens` 与总 token 数做一致性检查，将真实 ID 绑定到
执行线程。异步 host task queue 会破坏这一 TLS 假设，当前接入要求
`TASK_QUEUE_ENABLE=0`、eager、`async_scheduling=False`。多请求集合只能证明请求集合，
不能自动证明每个 MatMul 行对应哪条请求。蓝区验证器通过真实
`InputProcessor.assign_request_id` 观察外部/内部 ID 映射；黄区也需验证自己的真实入口，
无法关联时保留 `UNPROVEN_REQUEST_MAPPING`。

## 4. Native 部署边界

源代码与 ABI 定义见 [native/README.md](../native/README.md)、
[build.sh](../native/build.sh)、[fault39_api.h](../native/fault39_api.h)。
当前按实际 aarch64 CANN 安装的头文件／符号／运行库指纹构建，保留 C++ ABI0
和独立 Fault39 ABI v1（Config 368 字节）。已完成 8.5.1、9.0.0、9.1.0、9.2.0
CPU 构建预检；新适配版本的 NPU 验收仍待执行。不能把主版本号或编译成功当作
实机通过，详见[版本适配矩阵](cann_version_adaptation.md)。不覆盖历史 SHM v13，
也不复用其内存布局。

在实际 Worker 启动**之前**准备同一构建的 runtime 与 metadata proxies：

- `libfault39_native_v1.so` 和 `libopapi.so` 进入 preload；代理库优先于真实 opapi。
- 当前 torch-npu 2.9 经 handle-specific `libopapi_nn.so` 查找 MatMul，因此还需对应
  split proxy。蓝区 controller 将其只读映射到任务容器的匹配库入口；只修改搜索路径
  不足以证明实际捕获到了这一调用。
- `real/libopapi_real.so`、`real/libopapi_nn_real.so` 来自当前匹配安装，保持构建指纹。
  它们是环境私有依赖，不随源码或公开证据上传 GitHub。
- `CtypesProvider.attach_preloaded()` 使用 `CDLL(None)`；不得以运行中 late-dlopen
  代替 Worker 重启后的真实 preload 验证。

这里不要求修改共享主机库，不提供新服务部署命令。黄区应在自己的隔离运行环境中
完成相同加载关系，并重新验收 typed metadata、cache、真实 Run slots 和无损替换。
不能只凭 `.so` 存在或 counter 非零宣布 Native 路径通过。

## 5. 泛用性与拒绝条件

[target.py](../src/demoncat_fault39/target.py) 根据真实 MatMul 元数据选择目标；
[fault39_core.h](../native/fault39_core.h) 校验动态 M/K/N、BF16 ND、正偶数 K、
正且不重叠的 stride、offset/storage 边界和输入输出不别名。不是按模型名称套 Qwen 形状。

默认 `unique` 拒绝歧义；显式可选 `max_k_min_m_unique` 在合法候选中取最大 K、最小 M，
仍要求选中的签名/cache 唯一且跨工作负载稳定。相同 cache 可被多层共用，当前
`max_hits=1` 是作用域内首次匹配调用，不代表唯一层或全部层覆盖。

量化 QMM、`aclnnMm`、其他 dtype/格式、绕过此 API/cache 的编译或 fused 路径，均不能
继承 MatMul 结论。发现阶段应报告缺少受支持目标，不应改用 Python 输出扰动作为回退。
新模型需用真实权重重新发现、无损验证、注入和恢复；新 shape 不是第二架构的证据。
非零 offset / 其他 stride 的实机范围及 TP 范围以单独证据为准。

## 6. 现有蓝区工具的最小复核路径

这些工具使用现有 124 环境和离线模型，不启动常驻服务。先用
[package_source.py](../tools/package_source.py) 生成带 SHA-256 manifest 的独立源码包，
在匹配 CANN 环境按 `native/build.sh` 构建独立目录；保留源码和构建身份，不覆盖旧产物。

以下变量须指向本次已核实的绝对路径，`F39_RUN` 为新的实验根目录。示例只示范
现有 controller 参数；不授权停止其他任务。controller 持有共享锁，并在每个 case
前检查实际 NPU 进程和打开的设备句柄，最多执行 8 个 case。
模型命令之前必须先在新的 attempt 完成 P2、P3、P4；下方省略了仅更换
`--cases` 和 `--attempt` 的 P3/P4 重复命令。

```bash
python3 "$F39_SOURCE/fault39/tools/run_124.py" \
  --cards 5 --source "$F39_SOURCE" --native "$F39_NATIVE" --model "$F39_MODEL" \
  --attempt "$F39_RUN/p2" --cases "$F39_SOURCE/fault39/contracts/p2_cases.json"

python3 "$F39_SOURCE/fault39/tools/run_124.py" \
  --cards 5 --source "$F39_SOURCE" --native "$F39_NATIVE" --model "$F39_MODEL" \
  --attempt "$F39_RUN/model-lossless" \
  --cases "$F39_SOURCE/fault39/contracts/model_policy_cases.json"

python3 "$F39_SOURCE/fault39/tools/run_124.py" \
  --cards 5 --source "$F39_SOURCE" --native "$F39_NATIVE" --model "$F39_MODEL" \
  --attempt "$F39_RUN/model-statistics" \
  --cases "$F39_SOURCE/fault39/contracts/model_statistics_cases.json" \
  --lossless-receipt "$F39_RUN/model-lossless/model_lossless/model/validation.json"
```

按顺序补充 [p3_cases.json](../contracts/p3_cases.json)、
[p4_cases.json](../contracts/p4_cases.json) 的原生无损和五 pattern 门禁；
[model_discovery_cases.json](../contracts/model_discovery_cases.json) 保留严格发现结果；
[model_pattern_cases.json](../contracts/model_pattern_cases.json) 对完整模型运行五 pattern。
模型注入要求来自同模型、TP、目标和环境的真实 lossless receipt，不能复用别的模型回执。
双卡另用 [tp2_lossless_cases.json](../contracts/tp2_lossless_cases.json) 和
[tp2_pattern_cases.json](../contracts/tp2_pattern_cases.json)，并同时通过卡 2/5 门禁；
不能由运行命令存在推断实测已通过。

## 7. 证据与黄区验收要求

- 原生证据与 msprobe 原始三件套分开保留。Native 临时输入统计来自 host 构造，
  `input_device_readback_verified=false`；输出来自真实 Run 后 D2H，不得混称设备回读。
- `finite` 包含 `zero`；不同硬件的 Inf/zero 分布不是固定标准答案。合法命中但未出现
  非有限值时，保留 `RANGE_ANOMALY_NOT_OBSERVED`，不要修改输出制造结果。
- 原始 `statistics/mix` 保存 dump、stack、construct、配置和 collector identity。
  [compare_model_capture.py](../tools/compare_model_capture.py) 按实际 step 数字顺序比较，
  拒绝 B/F step 不匹配，使用固定显式容差并保留 null。不得用命中日志选首差。
- 被选请求结束后，必须先通过相同 owner 调用 Worker 的 `finish_capture`，确认
  `CAPTURE_FINISHED`，再执行 OFF 恢复请求。短 EOS 也必须立即结束采集，不能等待
  预设 step 数耗尽；结束采集失败时中止 OFF，不允许把恢复调用混入 baseline/fault。
  蓝区验证器在所选 `generate` 的 `finally` 中完成此检查，每个采集使用新 Worker。
- msprobe 记录调用方输入，而替换在 Python 下层发生，两者输入观测不同不构成矛盾。
  解码后续输入可能随生成结果变化；它们的差异不能直接归因于新的 Native 命中。
- 算子探针比较完整输出字节和输入存储哈希；模型门禁比较 Native 证据、token/text
  响应及恢复。响应相同不等于全模型中间张量逐元素相同。
- 公开交付只包含源码、原始统计、配置、版本和报告；不包含完整私有张量、模型权重或
  vendor 二进制。报告分别列出 CPU/mock、真实算子、真实模型、TP 与内源接线范围。
  2026-11-16本轮按用户要求仅提交实现与总结，新的原始statistics保留在124，不随Git发布。

黄区最终应重新验证：实际 rank 过滤及未选 rank 零替换、请求映射、同步 disable/drain、
重复启停、与另一条真实 fault 共存时的 owner 隔离，以及内源 CLI 到 Native 的闭环。
蓝区同一 Native ABI 的 foreign-owner 拒绝测试不能代替内源多 fault 集成验收；
正常 OFF 成功也不能代替强制超时/同步失败恢复验证。
