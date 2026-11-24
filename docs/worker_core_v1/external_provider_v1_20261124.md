# r8 窄外部 provider 接口

负责人 2026-11-24 明确批准开发及 A2 示例验收。接口不改变冻结 r7.1；外部能力不必加入公共静态机制目录。

## 边界与接线

可信部署代码负责导入自己的适配器，并调用 `external_provider.register_external_site`。
公共 CLI/RPC 不接受 Python 模块路径、代码字符串、裸地址或用户指定层名，不替部署方加载任意代码。
注册需要既有 DeploymentAuthority 的鉴权，完整实际部署身份、适配器模块源码 SHA、当前 model 对象一致。
注册本身不安装 hook、不发起推理、不取得资格。外部 factory 的构造函数必须无副作用；资源安装只在 `install()` 中执行。

机制名为 `external.<命名变体>`，不能覆盖现有内置机制。首版仅支持 `PRODUCT_ACCEPTANCE_ONLY`，
每次验收仍需同名、同 Worker 的临时许可；不能用 builtin 的许可借用，也不能申请 `PRODUCT_REGISTERED`。

可信 Worker bootstrap 中的调用骨架：

```python
from demoncat.runtime.vllm_ascend.worker_extension.external_provider import (
    ExternalSite, register_external_site,
)
from demoncat.runtime.vllm_ascend.worker_extension.product_binding import configure_product_profiles

# ProviderClass 是部署代码已导入、已审查的实际类型。
# expected_deployment 和 source_sha256 来自部署方的审查材料；不能由用户请求传入。
register_external_site(worker, ExternalSite(
    "external.named_variant", ProviderClass, source_sha256, expected_deployment,
), authority=authority, operator=trusted_principal)
configure_product_profiles(worker, profiles={"external.named_variant": {
    "mechanism": "external.named_variant", "mode": "PRODUCT_ACCEPTANCE_ONLY",
    "canary_prompts": [canary_token_ids],
}}, authority=authority, operator=trusted_principal)
```

所有 profile 应在一次 `configure_product_profiles` 中配置；不能覆盖已经安装的 profile。
完整部署身份使用 `deployment_acceptance.observe_deployment` 的字段集合，逐字段匹配，不能只提交 TP 等部分字段。
示例工具在隔离的蓝区临时验收中记录现场身份，凭限定 scope 的临时许可运行；它不产生发行登记。

可信宿主用 `ExternalFaultConfig('external', 'named_variant')` 构造普通 `ProductClient`。
随后仍是 attach / enable / status / disable / detach，宿主不传请求 ID。
`TrustedProductSession` 从各客户端动态 capabilities 选择机制；黄区主项目可把自己的 CLI 映射到这个 helper。
`FaultConfig.from_dict` 和公共部署 JSON 仍不接受 external 代码加载。未安装的外部机制返回拒绝，不表示该故障在目标部署上永久不支持。

## 适配器协议

factory 接收 `(model, runner, *, fault, region_leases)`，返回 site：

| 成员 | 责任 |
|---|---|
| `install()` | 自动发现真实模块/算子调用，安装本 scope 的 hook；不得自行驱动推理 |
| `finish_forward()` | 在真实 compute_logits 返回后退出观察上下文；确认设备消费者读完，区域租约清零 |
| `close()` | 释放本适配器持有的 hook；未读完或未恢复时拒绝成功关闭 |
| `region_leases` | 必须是公共框架传入的同一 RegionLeases 对象 |
| `events` | 每次调用的 request_ids、site、eligible/applied/consumed/restored/failed、changed、前后恢复哈希和实际消费者 |
| `forward_calls` | 每个实际 forward 的调用计数，公共收集器核对与 events 相等 |
| `sites / pending / uncovered` | 声明范围、未完成的消费者和未覆盖调用；不能静默丢弃 |

布尔计数必须为真实 bool；changed 为非负整数，未 applied 的事件不能宣称 changed。
事件 request_ids 必须等于当前由 Worker 发现的请求，不能借用基线清单。
同 stream 的设备完成和字节恢复复用 RegionLeases；不能用 Python 返回代替设备完成。
公共框架负责八项计数、每 rank 汇总、Runtime 租约、绑定互斥、实际调度请求映射、超范围拒绝及金丝雀。
适配器属于可信部署代码，源码审核仍是必要条件；这是接线协议，不是运行不可信代码的沙箱。

## 首版范围

- 沿用已审核的 0.26 runner/model 调用边界，serial text、单 chunk、fresh cache、context≤1024、1–4 生成 token。
- 全 rank 持续触发。不同 rank 的适配器源码 SHA 必须一致；每个 rank 的实际部署身份分别校验。
- 未扩展多模态、Batch、EP、黄区自有 sampler/HCCL 路径；这些需要各自适配器和独立验收。
- 生命周期完全复用 ProductProvider / AutomaticRequests / WorkerFaultRuntime，不另建状态机。
- A2 示例为 `external.gain_three_quarters`：源绑定的真实 norm 返回区域缩放 3/4，再由实际消费者使用；不是黄区十六项的代理验收。
- CPU 测试覆盖缺许可、错误源码、身份漂移、模型替换、builtin 覆盖、用户代码路径及资源互斥；实机结果见同目录结果记录。

## 黄区下一步

负责人转交本协议及候选实现，黄区负责自己的 provider 选点、注入、消费者和安全恢复。
先确认自然调用链，再做两个 owner、全 rank 计数、真实消费者、区域恢复/资源零/部署金丝雀三项恢复及负例。
未接线、未观察和不适用分别记录。公共 catalog 中没有条目不能单独推出“不支持”。

控制面约定：binding/可信 bootstrap 的预期拒绝返回结构化 exit_code=4；内部错误为 5。正式模型 forward/kernel 异常不在此拦截范围。enable 失败由原 Runtime 按 owner 清理，未确认清零的失败保留恢复责任，可再次 disable 重试。
