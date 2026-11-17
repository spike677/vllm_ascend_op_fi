# E2 公共 Native Runtime 和 Observer

`native_observer.py` 统一接收 DynamicQuant V2 和 Fault26 的现有独立 ABI 原始数据。`NativeObserver.capture(scope)` 在被观察的调用之前读状态，在调用返回以后验证增量。已有 `NativeDynamicQuantInventory`、`NativeWriterInventory` 校验 ABI、容量、序号、计数一致性和丢失事件。结果保留每个原生事件、拒绝或未完成的调用、跨线程调用、原生线程和 Device、Raw Tensor 地址和几何，以及 Executor、Cache 和注册生命周期原始字段。相同地址复用生成独立调用记录。

```python
from engineering_closure.native_observer import NativeObserver, NativeScope

observer = NativeObserver.dynamic_quant(pinned_quant_runtime)
scope = NativeScope(process_id, process_start_identity, worker_id, rank, logical_device,
                    request_ids=actual_request_ids, phase='decode',
                    dispatcher_token=actual_dispatcher_token, request_observed=True)
with observer.capture(scope) as window:
    run_the_original_dispatch()
```

`NativeObserver.writer26` 直接封装已装载的只读 Writer/Consumer 日志。该 ABI 仍用 `correlate_lifetimes` 对真实生产者、Writer、下一 Native Consumer 及分配代号对账。DynamicQuant 仍用 `verify_public_native_call` 和 `SharedScaleInventory` 对实际输入、Q/Scale、QMM 与请求行对账。Native 窗口覆盖已观测 ABI 的分母。没有前置事件或查询被拒绝的调用保留为未绑定。通过实际 op API 归属和语义检查后，现有 `IndependentCallObserver`、`CandidateSet` 和 `CoverageSession` 继续负责目标注册、触发、生命周期、Drain、`TargetLease`、OFF、Worker 健康，以及效果/恢复摘要。Fault20 L4 Buffer Proxy 和 Fault26 Writer Proxy 保留现有原生 Adapter 与调用路径。

Linux 上 `PinnedNativeLibrary` 默认用 `RTLD_NOLOAD`，通过绝对路径、SHA-256、`dladdr` 和运行时映射的 inode 校验符号真正来自该映像。可显式选择已审查的主动加载。`preload_environment` 生成子进程环境副本，可用于既有部署控制器。加载与实际进入是两种独立证据，`symbol_entry_status` 只接受实际函数入口的地址和映像身份。能观察到真实 typed enter/exit 的路径可以直接建立调用图；绕过普通动态符号的 Dispatcher、ATB 或私有接口保留其独立 Adapter。

窗口采集和符号归属不会授权注入。同步同线程窗口需要 `TASK_QUEUE_ENABLE=0`、相符 Device 和完整生命周期才能绑定到 Dispatcher；诊断错误、异步调用、跨线程和未覆盖 ABI 都保留可见性边界。数值变化、真实 Consumer 使用和恢复分别由原有机制 Adapter 和独立证据审计确认。

公共覆盖会话沿用 `once`、`every_match` 和实际目标 eligibility 条件。任意指定第 N 次调用的通用触发配置仍未统一实现，需额外定义其分母、作用域和部分覆盖回执。现有 C2/C3 规则保持各自触发语义。
