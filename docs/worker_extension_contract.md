# 共用 Worker Extension：隔夜接线合同与边界

基线为 `bfdd6abc1adb5d08b034db7a96ec97ffbec0b9bf`。继续使用
`demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension`，没有新增 Registry、
生产 CLI 或 Worker 类。vLLM-Ascend 保有 Worker 的创建、设备初始化、模型加载、
调度和退出职责。扩展只增加已有的六个 `demoncat_*` RPC 方法；不覆盖宿主核心方法。
本轮新增临时验收程序使用 vLLM 标准启动流程，由外部资源 controller 控制设备和超时。

## 身份与生命周期

每次 RPC 重新读取实际 PID/process start identity、Worker start token、rank、
已初始化的逻辑 NPU 和存活 model instance。没有调用 `set_device` 来制造身份。
附着租户获得随机 nonce；prepare 获得单调 generation 和内部 native owner。
不同进程、模型实例、tenant、owner、generation 的票据不能互换。
Worker 重建不会自动 re-arm 旧故障；模型变更且仍有资源时，整个 Worker 进入不可复用状态。

生命周期仍为：

```text
identity → attach → prepare → register → check → execute（最多一次）
                                                      ↓
                           query ← finish / disable / abort
                                                      ↓
                                            clean → unregister → detach
```

execute/enable 的尝试在派发之前即被消费。异常、中断和未确认的结果都不能再次执行。
每个动作结束后读取实际 Provider 状态，不能只相信返回的成功字符串。
关闭要求 `enabled=false, inflight=0, leases=0`；释放还要求 native owner 已清除。

## 本轮修复：恢复失败隔离整个 Worker

原实现只阻止同一 lease 的后续故障动作；另一个机制或租户仍可能在恢复失败后创建
新任务。不同机制可能共享 NPU stream、allocator 或进程级 interposer，因此这一点
不足以满足恢复约束。

现在有任一 lease 未恢复时：

- 阻止新 tenant、其他机制 prepare 及已有其他 lease 的故障动作，返回 `WORKER_RECOVERY_REQUIRED`。
- 保留 query、身份、能力查询和原 owner 的 disable/abort/clean/unregister/detach。
- query 发现 native 状态缺失或未排空时也保留隔离；随后某次只读查询恢复为 idle 不自动清除隔离。
- KeyboardInterrupt 等非普通异常继续向上传播，但先记录恢复义务，避免存活 Worker 被错误复用。
- 恢复一个 Provider 不会清除另一个 Provider 的恢复义务；所有未决 lease 均完成显式恢复后才重新准入。

该隔离只停止扩展发起的新注入，不接管 serving 的健康管理。真实 native 无法恢复时，
宿主负责将临时验收 Worker 下线；扩展不会重启 Worker 或全局 reload 模型权重。

## 已接线的机制与实际门槛

| 机制 | 共用入口 | 执行前必须提供的真实条件 |
|---|---|---|
| 20 WeightNz | 已接线 | 实际 `aclnnMatmulWeightNz` self/slot、隔离构建、完整模型排他 replay、唯一稳定目标、reference/lossless/OFF |
| 20 QMM | 已接线，独立子路径 | 实际 `aclnnQuantMatmulWeightNz` x1/slot；不能沿用 WeightNz 的 slot 证据 |
| 39 matmul_range | 已接线 | 独立 ABI v1、匹配 profile、实际 scheduler request ID 与 self/mat2 全 storage 哈希、排他 Worker replay |
| 其他机制 | 保留既有独立实现/实验入口 | 尚未因本扩展的 prepare 成功取得通用 Worker 接线验收；专属 Provider 与恢复合同需要逐项适配 |

`LD_PRELOAD` 必须由 serving 启动环境提前设置。prepare 只能连接已经加载且身份匹配的
runtime，不能在 attach 之后用 `dlopen` 补建预加载事实。F20/F39 的多个库组合也不能
因分别通过就直接认定可同时部署；SONAME/符号/初始化顺序仍需独立验证。

F20 部署使用 `register_worker_matmul(..., replay=..., reserve_worker=..., profiles=...)`。
回放必须执行恰好一个完整的已加载模型 forward，返回完整输出 tensor，而非token摘要。
QualifiedNativeSession 不支持“任意挑一个层”：同 profile 发现多个稳定目标仍为
`AMBIGUOUS_TARGET`。它没有 `max_k` 选择规则，本轮不通过增加隐藏层选择绕过单点资格。

F39 部署使用已有 runtime registry 的 `Fault39WorkerReplay`，每轮返回
`ReplayObservation`。当前 `QualifiedFault39Backend.enable()` 在单一排他 RPC 内执行
两轮发现、三种无损、单次故障及 OFF。每轮需要不同的真实 scheduler request ID、
一致完整输入/原存储哈希和完整模型输出哈希。一次 model.forward 捕获再多次重放不能
自动证明这些新请求；由调用方填字符串或复制历史请求号不满足合同。
独立 Fault39 runner 的历史 NPU 结果也不证明此共用入口已完成真实 scheduler 接线。

### 真注入仍缺少的部署接口

本轮实际拒绝与代码位置、CPU 合同结果及下一步设计见
[worker_injection_gap.json](../experiments/overnight_20261116/worker_injection_gap.json)。
共用调用链为 `demoncat_fault_rpc → WorkerFaultRuntime._run → Provider.dispatch`。
RPC 在 Provider 返回前同步占有扩展锁；该锁只串行化扩展 RPC，不能代替宿主的请求排他预约。

- F20 的 `providers/fault20.py::_register` 读取 `qualified_binding`，要求部署通过
  `register_worker_matmul` 提供真实模型身份、已审查 profile、`reserve_worker` 和完整模型 replay。
  当前生产代码没有安装这份绑定，因此返回 `FAULT20_TRUSTED_EXCLUSIVE_REPLAY_REQUIRED`。
  `QualifiedNativeSession._replay` 要求恰好一次完整已加载模型 forward，
  `_digest_output` 要求完整非空 tensor，后续继续核验稳定唯一目标、原 storage、无损及 OFF。
- F39 的 `providers/fault39.py::_register` 要求 `Fault39WorkerReplay`，当前缺失时返回
  `FAULT39_TRUSTED_SCHEDULER_REPLAY_REQUIRED`。
  `host_qualification.py::enable` 在同一次排他预约内同步执行七个阶段；
  `_workload` 拒绝重复 request ID、原 storage 前后不等或基线不一致，
  `_observe_run` 要求 native 记录的实际单请求集合精确等于本次 scheduler ID。
  当前未实现能在该 Worker 的同步 RPC 执行期间推进真实新调度请求的宿主 replay 接口。
  普通 driver 在外部调用 `generate` 不足以补齐这项能力。

F20 的最小下一步是新增部署拥有的 replay 适配：从真实请求通过可移除的模型 hook
捕获完整参数、attention metadata、forward context，并制定 KV 状态重置规则；
在排他预约下、同一模型线程重放，返回完整确定性输出。F20 不要求每轮新 scheduler ID，
因此可以先验收“实际已加载模型的捕获回放”，但不能将其写成新的 scheduler 请求验收。
任何多目标或未证明调用仍拒绝，不能选第一层、最大 K 或隐藏地缩小层范围。

F39 首先需要宿主侧可信的 scheduler replay 边界，将真实新请求与模型线程上的
`provider.context`、完整输入/输出和 self/mat2 原 storage 读回绑定。
如果部署引擎不能同步重入，就需要另行审查分阶段 continuation 编排；
那是接线协议的后续实现，不能只增加注册函数或复用旧请求号来完成。
后续设计仍须保留一个尝试贯穿全阶段、唯一目标、单次命中、owner/generation、
失败隔离和 OFF 恢复，不新增 Registry/CLI，不覆盖 Worker 核心方法。
历史 `fault39/vllm_worker.py` 的 runner wrapper 是另一条入口，不在本次共用扩展中启用。

### 本轮真实发现的有限结果

TP1 的正常 serving WeightNz 观察完成两次 forward，113 个结构候选共226次模块调用，
native 事件为0，实际状态 `NATIVE_OPERATOR_EVIDENCE_REQUIRED`。
这表示当前 serving 路径没有取得受支持 WeightNz 调用证据，不能推断算子不存在或注入成功。
controller 已设置 `TASK_QUEUE_ENABLE=0`，Python 与 C++ 安装入口也都检查该值；
没有证据将零事件归因于缺少该变量或线程错误。本轮保留此结果并停止重复尝试。

实际32B W8A8模型的 TP2 QMM 两个 rank 都完成两次 forward。
每 rank 258 个结构候选、514次模块调用，只得到3个真实 native 事件：
全部属于 `forward=0` 的第0层 `qkv_proj`、`o_proj`、`gate_up_proj`，
`forward=1` 为0；其余511次模块调用没有 native 事件。
真实报告的资格状态是 `PROFILE_REQUIRED`，没有已绑定 profile，也没有选定目标。
三种签名不是“三个稳定候选”，更不是已满足唯一目标；两轮稳定观测尚未成立。
可能的缓存命中分支或实际 API 覆盖问题需要单独调查，本轮没有认定根因。
这两项发现均无故障注入；观察前后正常生成保持相同，回调已释放，native idle。
公开摘要保存原始私有报告的 SHA-256、阶段计数和未观察范围，不发布私有报告正文。

### 中断与恢复复核

本轮新增的取消/恢复处理未发现新的严重缺陷：execute 派发前消费尝试；
Provider 动作或 readback 中断先保留恢复义务，再继续向上传播；RPC 锁由 finally 释放。
恢复动作仍需新鲜 native idle/inert 读回；detach 部分失败不删除未决 lease 和 tenant。
复核使用现有6项 Provider、12项资格适配、4项 Fault39 host、19项 Worker CPU 合同测试，
全部通过；它们使用明确的模型/原生测试替身，不是实机故障命中证据。

隔离不拦截普通 serving 的 `execute_model`。宿主必须阻止恢复未决的 Worker 再接收请求；
客户端 RPC 超时也不证明 Worker 侧执行已结束，不能据此立即复用设备或重试故障。
进程被强杀时 Python finally 无法执行，进程清理仍由宿主负责；旧身份票据不能复用。

## 本轮可执行验证

[`worker_probe.py`](../experiments/overnight_20261116/worker_probe.py) 复用历史 probe 的
身份、RPC 与原子回执写入，没有修改旧实验文件。它提供两类有限目标：

1. `quarantine-fault20` / `quarantine-fault39`：在临时 TP1 Worker 上验证匹配预加载、
   缺少可信 replay 的注册拒绝、跨机制隔离、原 owner 释放以及正常生成恢复。
   即使通过也只能写“真实加载/隔离/恢复通过”，不能写“故障已命中”。
2. `observe-fault20-weightnz` / `observe-fault20-qmm`：通过实际已有 Worker RPC 安装
   read-only NativeSession，正常 generate 两次相同 prompt，移除并核对原生 idle。
   记录真实 API、slot、模块、cache identity、调用计数和未观察到的分支。无损资格及注入另行验收。

Observer 安装绑定当前 Python thread。已有 NativeSession 会拒绝异线程模型 root，
ctypes callback 只接受同一 thread 的实际 module scope。回执记录安装/结束 thread ID；
`same_thread_observed=true` 还要求实际 native 事件非空及已完成的模型 forward。
仅成功安装或 RPC 返回不能证明观察到了同线程模型调用。

示例入口由外部 controller 预先完成资源锁、空闲门禁和只读 bundle 挂载后执行：

```bash
python /source/experiments/overnight_20261116/worker_probe.py \
  --case quarantine-fault39 --tp 1 --model /model --out /evidence/result.json
```

`--memory-utilization` 可设置为 0.1–0.85，默认 0.35。实际 32B W8A8 TP2 QMM
发现测试使用 `--case observe-fault20-qmm --tp 2 --memory-utilization 0.75`；
改变加载预算不改变 Provider 或单目标资格门槛。

CPU检查为 24 项 Worker/探针状态机测试与 6 项 Provider 桥接测试；
见 [worker_cpu_receipt.json](../experiments/overnight_20261116/worker_cpu_receipt.json)。
这些使用明确的 Host/Native 测试替身，不是新 NPU 证书。真实结果由隔夜最终报告另行引用。

新增 [C++/ctypes 回执](../experiments/overnight_20261116/worker_native_cpu_receipt.json)
在124的独立 `runc / network none / 无device` CPU容器完成。它验证原生回调线程隔离、
唯一continuation、两种真实C ABI形态的主机fixture、输入/slot/cache代次不匹配拒绝、
无损/OFF，以及同步失败后的资源保留。首次尝试因已有头文件的缩进warning被Werror阻断，
第二次保留该warning可见且不修改已验收Native源码；其他Werror与断言均保持。
两次容器均已移除，原始尝试保留，实际NPU调用数均为0。

已审查的F20构建为124的
`/home/worker-extension-20261116-a1-a3bb5996/sdk-output/build`，profile
`2fd278ad333a5d37060fbfb017eabacfb95c8ba4f6796e796a89d2feb8c8eae1`。
[15条源码核验](../experiments/overnight_20261116/fault20_bundle_review.json)
均与 `bfdd6abc` Git blob一致，已有文件完整性校验也通过。该核验只确定构建来源，
不把新统一Worker的故障闭环自动升级为通过。

原始 statistics/mix 和新 Worker 回执留在124；公开部分只提交实现、结果摘要、路径与哈希。
负门禁案例不制造名为 fault 的健康采集来冒充 B/F 数据。
