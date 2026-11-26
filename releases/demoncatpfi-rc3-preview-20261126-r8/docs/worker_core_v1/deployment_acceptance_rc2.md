# RC2 部署验收许可与登记

状态：实现与 CPU 契约验证；黄区 RC2 的组合记录当前为空。用户报告 RC1.1 的 18/39 各两个作用域已通过，但没有 RC2 实际部署字段与新回执，不能直接登记为 RC2。

部署身份为原 13 项字段加 `hardware_family`：torch、torch_npu、vllm、vllm_ascend、CANN 版本列表、dtype、TP/PP/DP、eager、实际 model_class、config.json SHA256、live_config_matches_file、A2/A3。硬件只查询已经初始化的 NPU；未知为不可观测，不猜测。蓝区既有组合登记 A2。硬件 family 不代替实际部署摸底中的具体芯片型号。

来源隔离：`BLUE_COMBINATION`、`YELLOW_TEST_DEPLOYMENT`、`YELLOW_DEDICATED_INSTANCE`。未配置来源默认关闭；黄区不能借用字段相同的蓝区记录。下一发行版中人工审核登记 `reviewed_deployments.YELLOW_RECORDS`，许可与总结工具均不写这张表。

## 部署侧接线

在实际 Worker 的可信初始化代码中绑定一次。这里的权限回调必须来自部署操作员的认证/授权系统；不接受 FaultConfig、普通用户参数或用户传入的回调。相同进程中任意 Python 本来就能修改 Worker，此 API 不声称抵御该权限。

```python
from demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization import (
    configure_deployment_acceptance, install_deployment_acceptance_permit,
    remove_deployment_acceptance_permit,
)
authority = configure_deployment_acceptance(
    worker, source="YELLOW_TEST_DEPLOYMENT", authorize_operator=deployment_authorizer,
)
permit = install_deployment_acceptance_permit(
    worker, authority=authority, operator=authenticated_operator,
    mechanisms=["18.api_half"], attempt=unique_attempt,
    source_manifest_sha256=installed_source_review_sha256,
)
# 沿用真实 binding + Host attach/prepare/register/check/advance/wait/query，
# 实际 normal/fault/off/health，finish/unregister/detach。
remove_deployment_acceptance_permit(
    worker, authority=authority, operator=authenticated_operator,
)
```

许可绑定机制、仍存活 Worker/model 对象、pid/thread、attempt、清单 SHA；每次动作和 query 回执包含 `deployment_acceptance_use`。作用域改变时拒绝新动作，但允许恢复；仍有 owner 时不能撤销许可。操作员授权回调分别检查 `install_acceptance_permit` 和 `remove_acceptance_permit`。

清单 SHA 是可信部署侧提供的绑定身份，**不是**仅凭一个 SHA 字符串就证明已安装源码；仍须执行发行包的实际安装源码校验，以及各 Native/source/scheduler/live-call 检查。许可不能打开未实现能力，不能制造 accepted 记录。来源配置本身也不授予验收许可。

## 文字回传

保留两次独立 owner 的最终 Host `receipt()`（detach 后）。每个完整回执最多 16 MiB，按以下命令导出不超过 64 KiB 的紧凑 JSON：

```sh
python tools/summarize_deployment_acceptance.py --scope scope1.json --scope scope2.json --out acceptance-summary.json
```

总结包括部署字段、来源/许可身份、两作用域 Worker/owner、请求阶段及效果、恢复、资源状态和原回执 SHA。不传权重/完整张量；projection 字段缺失、不可观测、资源未清零、来源/模型不同、重复 owner 或超限都明确标记。`review_ready` 只表示材料可供审查，**不代表机制验收已通过**。原回执不是签名证明，需要部署操作员对真实性负责；Codex 核对请求效果、恢复、资源以及源码身份之后，才在下一发行版本登记。

请另附具体硬件型号、发行源提交/清单 SHA、三个 wheel SHA、真实安装源码校验结果，以及机制自己的适用范围。旧回执缺新增来源/硬件字段时不得补造默认值。默认无许可的已登记组合也逐动作保留来源身份。

RC2 黄区清单仅 18/39/22；8/17 保留蓝区验收；Batch 黄区待配置。升级 RC2 后，黄区 18/39 在组合登记前默认关闭；使用上述授权许可验收回传后，由 RC2.x 登记恢复默认准入。RC1/RC1.1 包与标签保持不变。

Batch 的前后健康请求在六批 Host 协议之外；仅提供 Host 回执时总结标记 `BATCH_EXTERNAL_HEALTH_RECEIPT_REQUIRES_SEPARATE_REVIEW`，必须另审真实健康回执，不能将 Native recovery_proven 冒充健康请求证明。

## 冻结后补充：许可清单与审查（2026-11-19夜）

RC2 的 `source_manifest_sha256` **填写发行目录中 `source-review.json` 原始文件字节的 SHA256**：
`e1622a4bdc91f31576342e9d66e02f75e092f41738bb8fe8b8413fe553aecdea`。
不是 `release-manifest.json` 的 SHA256（后者为 `d8369cba27b5d7a5dc4e460123fdfa89550fbbd231d7082cab9c79abc505c72b`），也不是 tar.gz、config.json 或单个 wheel 的 SHA。

审查从冻结标签 `demoncatpfi-yellow-rc2-20261119` 获取清单：逐文件验证 SHA256SUMS，包括 source-review、release-manifest、源包、三个wheel及安装回执；对source-review原始字节重算SHA，与上述常量、许可和每个作用域回执三方核对。再核对 source-review 与源提交 d07c72156e0fe4e3608dcee58cdb422b49834c66 对应的源文件清单、wheel安装后的实际模块路径与源码字节（蓝区离线核验285个已安装Python文件）。黄区须给出自己的安装核验，不沿用蓝区PASS。API本身只绑定SHA字符串，不读取文件，不能把许可安装成功当作源码验证。

## 冻结后补充：普通 binding 对照

前缀均为 `demoncat.runtime.vllm_ascend.worker_extension`。只在可信Worker初始化中安装，FaultConfig不接收这些权限或构建参数；不创建Worker。

|能力|模块、安装函数|必填参数和普通入口限制|移除函数|
|---|---|---|---|
|18.api_half|`serving_tensor_binding.install_rms18_task_binding`|`worker, prompt_token_ids, weight_manifest, workload_sha256, environment_sha256`；默认生产参数，无_simulation；内部固定runtime_only=True|`serving_tensor_binding.remove_tensor_serving_binding(worker, "18.api_half")`|
|22.wrong_row|`serving_tensor_binding.install_rope22_task_binding`|`worker, prompt_token_ids, weight_manifest, workload_sha256, environment_sha256`；默认生产参数，无_simulation；内部固定runtime_only=True|`serving_tensor_binding.remove_tensor_serving_binding(worker, "22.wrong_row")`|
|39.output_proxy|`serving39_output.install_fault39_output_serving_binding`|`worker, transport, prompt_token_ids, runtime_only=True`；transport为核验过的NativePostwriteLibrary；保留stage='c1'、max_live_leases=1、evidence_dir=None|`serving39_output.remove_fault39_output_serving_binding(worker)`|

先完成实际OFF/health、finish/unregister/detach，确认owner/Native lease/hooks清零，再移除binding与许可。39的transport构建与SHA依赖见rc1_native_dependencies.md；不拿蓝区SO路径当黄区制品。

## 冻结后补充：统一宿主调度语义与已知差异

初始化 attach/prepare/register/check/enable 后，读取真实query；每轮 **enqueue → advance(实际内部ID) → 等待真实完成 → query**，读取 `next_phase`，直到其为 `complete`，然后 finish/disable/unregister/detach。请求数由provider状态决定，不按故障号写死循环次数。advance只绑定将发生的真实调度，不代跑模型；必须在该请求执行前绑定。异步宿主应使用已有排队/调度边界，不允许enqueue后放任请求先跑完才advance。请求失败、证据超限、未知/缺少phase均进入失败/恢复，不猜下一阶段、不重试F。

以下路径相对于Runtime回执内的 **provider状态对象**（宿主应按真实RPC返回解出provider，不把Host外层action日志当provider）。静态核对到冻结RC2源码；41为开发线：

|能力|next_phase路径|完成标志|正常阶段序列|
|---|---|---|---|
|8/18/22|`serving.next_phase`|`serving.complete`并核对恢复/资源|normal → fault → off → health → complete|
|17|`serving.next_phase`|`serving.complete`并核对恢复/资源|normal → fault → off → health → complete|
|39 output_proxy|`next_phase`|`evidence.complete`并核对恢复/资源|normal → fault → off → health → complete|
|41（仅开发线）|`serving.next_phase`|`serving.complete`并核对恢复/资源|normal → normal_repeat → lossless → fault → off → health → complete|
|Batch|`serving.next_phase`|phase=complete，另核对`serving.recovery_proven`及前后健康|前健康 → normal → reference → lossless → calibration_off → fault → off → complete → 后健康|

核对结论：所有已实现普通任务在正常、未超限query均有next_phase，但不是统一位置；Batch没有统一complete布尔值，健康请求位于六批协议外。Batch每轮一次enqueue提交匹配B8并以 `advance(request_ids=[内部ID...])` 绑定8个ID；按其next_action及原协议保留需要的decode/collect控制，不能用普通单ID替代；query只读，不用query推进采集。阶段数量表是现有协议说明，不是硬编码调度算法。

RC2.2接口修正项：统一next_phase/complete及失败/恢复状态投影；在证据超限分支仍可读取明确失败终态（现有bound_state会省略phase，不能误判完成）；补Host统一协议测试；统一Batch外部健康回执。**这些问题本次仅记录，不改RC2源码/标签/包，不将未实现的统一返回写成现有保证。** 20/26/33/28未实现普通任务不在“全部有next_phase”的范围。

黄区当前不启动RC2验证任务、不向openchamber发送内容。master继续RC1.1；等上一轮回传与本补充单齐备，经负责人安排才在独立环境验收RC2。RC2.1只登记黄区组合，收到并审查通过前不建维护分支；新增41与接口修改另属RC2.2。
