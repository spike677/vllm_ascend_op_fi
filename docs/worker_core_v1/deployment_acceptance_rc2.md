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
    source_manifest_sha256=installed_release_manifest_sha256,
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
