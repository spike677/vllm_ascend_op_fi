# Serving Worker 附着合同

合同适用于通过 `--worker-extension-cls demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension` 加入已存在的 serving Worker。宿主持有 Worker 和模型生命周期。扩展持有本进程租户状态，调用现有 Provider，并报告实际状态。

## Worker 身份

以 Worker 当前 PID、进程启动标识、实际 rank、当前逻辑 NPU、已初始化的模型对象和模型实例 ID 为边界。Linux 下从 `/proc/self/stat` 与 boot ID 得到进程启动标识，不可读时使用带 PID 的进程内随机标识。rank 来自 Worker，设备与已初始化的 NPU context 对照。visible devices 记录从 `ASCEND_RT_VISIBLE_DEVICES` 或 `ASCEND_VISIBLE_DEVICES` 读到的设备列表和来源。缺失或无法解析的列表保留为空。模型实例 ID 绑定实际 runner/model 对象，同名模型替换也需要重新 attach。不能可靠观测关键值时拒绝建立故障租户。

Worker 身份同时记录 `worker_start_identity`、实际 Worker 类、vLLM 与 vLLM-Ascend 版本和扩展版本。租户票据绑定 `tenant_id`、随机 `attach_nonce` 与完整 Worker 身份。作用域继续用 `mechanism`、`implementation_version`、`owner` 和 `generation`。进程重启、模型替换、票据失效或原代次被清理时，旧控制请求必须停止。新 Worker 由宿主管理，客户端重新取得身份、附着、注册和校验。

## 扩展 RPC

| 方法 | 输入及语义 |
| --- | --- |
| `demoncat_worker_identity()` | 返回当前 Worker 的实际身份与可观测性。就绪时返回 `status=WORKER_IDENTIFIED` 以及 `worker` 对象。 |
| `demoncat_capabilities()` | 返回扩展与已接线的 Provider 描述。此方法不解析部署绑定或访问 Native。 |
| `demoncat_attach(payload)` | 比对 `tenant_id` 和 `worker`，附着已初始化的 Worker。成功返回 `status=ATTACHED` 和 `ticket`。 |
| `demoncat_fault_rpc(action, payload)` | 使用票据与机制作用域，通过本地 Provider 执行动作。 |
| `demoncat_fault_query(payload)` | 以同一租户与代次查询实际 Native 状态。 |
| `demoncat_detach(payload)` | 解除本租户附着。存在尚未确认回收的作用域时，保留租户和该状态供核对。 |

在附着票据中加入 `mechanism`、`implementation_version`、`owner` 后执行 `prepare`。响应中的 fault ticket 增加 `generation`，后续机制动作与查询传回该票据，detach 仍用最初的附着票据。Provider 参数放在 `settings` 下，按所选实现及 profile 校验。其他 rank 收到定向请求时返回 `RPC_NOT_TARGETED`。本 rank 继续核对完整身份。Controller 应验证目标响应唯一且与指定 Worker 一致。

通过 `prepare` 后才能 `register`、`check`，随后允许一次 `execute` 或 `enable`，二者共用一次尝试额度。结果存疑的执行也消耗该额度。当前两条接线需要部署提供可信、排他的回放能力，并在相应机制内执行自动选点和无损门禁。准备并查到空闲 Native 只完成附着验证。停止和回收动作包括 `finish`、`abort`、`disable`、`clean` 和 `unregister`，分别按当前状态及机制检查。

## 事务和 Native 状态

状态归属限定为本 Worker、本租户、当前 owner 和代次。未确认清理的租约继续占有对应机制。完成路径由 Provider 先停止新增干预，再依据机制核对在途调用、实际状态、缓冲区与原始配置。清理成功需要真实 Native 回执。查询失败明确返回 `NATIVE_STATE_UNAVAILABLE`，并保留相应的未知状态。查询展示的登记、资格、可观测 hit、OFF 和恢复是独立证据。如果持有机制租约时实际设备或模型身份变更，扩展冻结对该 Worker 的附着和控制，由宿主处理该 Worker 的回收。

现有 serving 的 RPC 用于命令和摘要。张量和原始采集留在 Worker 的验证路径。多个 Worker 的调度和失败聚合由宿主完成，各 Worker 各持其实际票据及 generation。票据限制进程内归属和代次，RPC 访问权限由宿主控制。初始验收范围见兼容性记录。

[扩展 RPC 源码](../../src/demoncat/runtime/vllm_ascend/worker_extension/extension.py) · [Worker 租约与状态机](../../src/demoncat/runtime/vllm_ascend/worker_extension/runtime.py) · [Worker 身份](../../src/demoncat/runtime/vllm_ascend/worker_extension/identity.py)

[架构及已有入口对照](../worker_extension/architecture.md) · [部署与最小探测](../worker_extension/deployment.md) · [实机范围](../worker_extension/compatibility.md)
