# E1 模型执行路径发现

`path_discovery.py` 复用 `inspect_structure`、实际 Worker 进程身份和 `NaturalTensorObserver`。结构角色用于标注 Attention、MLP、Norm、MoE 等模块候选。实际模块 pre-hook 和 Torch Dispatcher 调用提供执行边。记录有界的标量参数、输入和输出的 Shape、Dtype、Stride、设备、Storage 实例、Offset，以及 NPU Format 可读取时的值。观察窗口只读张量元数据，算子调用一次，原输出从同一个调用返回。若下一调用使用同一个存活的 Tensor 对象，会保留直接的 Output → Consumer 边。

```python
from engineering_closure.path_discovery import discover_execution_path

output, graph = discover_execution_path(model, lambda: model(**request_inputs))
```

模型本地调用带 `MODEL_LOCAL` 身份。普通 vLLM 请求可以通过 `worker_path_start` 安装观察器，在驱动侧按已有流程提交请求，最后同线程调用 `worker_path_finish`。预期请求 ID 来自实际驱动入队记录，Worker pre-hook 再从调度器验证一次。`worker_request_probe` 直接使用已有 `snapshot_fresh_prefill` 和 `snapshot_first_decode_rows`。每次 forward 分开记录请求集合、phase 及证据边界，保留 process start、Worker start、Rank、Device 和模型实例身份。现有读器覆盖受限的 fresh prefill 和 first pure decode。调度读器拒绝的范围在图中标为 `NOT_VISIBLE`。

`known_native_observers` 按实际 Dispatcher 名称精确选择已装载的独立 Native 读器。一次同步调用前后分别读计数器，以 dispatcher token、进程、Worker、Rank、Device、原生线程及 Native call/registration 序号绑定，保留窗口内未绑定的调用。KV Storage、Slot 和后续 FIA Consumer 的关联由已有 Fault41 Adapter 独立验证。离线转换可接收单独采集的 `KernelLaunchBoundary`。调用者提供的索引只能形成 `INFERRED` 连接，Host Tiling 计算及 Launch 的调用级因果链需要实际关联证据。

| 标记 | 证据含义 |
| --- | --- |
| `OBSERVED` | 实际模块、Dispatcher、同步 Native 窗口、返回或由专用读器提供的调用绑定证据 |
| `INFERRED` | 基于结构、源码或配置的假设。Native 路由提示需要源码路径和 SHA-256 |
| `NOT_VISIBLE` | 当前读器无法关联该边。例如融合算子内部、ATB 私有调用、异步线程、实际 Kernel Entry 或未被支持的调度范围 |
| `NOT_EXECUTED` | 已完整采集的窗口里已注册模块未进入，或已配置的同步 Native ABI 没有记录到事件 |

`normalize_execution_path` 也可只读转换已保存的 `demoncat.natural_tensor_census.v1`。旧报告保留其原始可见范围，详细操作事件少于 `operator_counts` 时可直接检查 `all_dispatch_details`。转换使用输入制品的原有可信度。`max_events` 控制本次诊断调用的上限，超预算会中断诊断运行。需要保留不完整报告时使用 `observe_execution_path` 上下文，退出后读取其中的 `report`。`discover_execution_path` 的运行验证目前为 CPU eager，Worker/CANN 集成接口继承原有读器的审查范围，跨图执行、其他调度协议和 Device Kernel 因果关系仍需要对应的运行时证据。
