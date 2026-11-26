# r7.3 外部 provider v1：安装与适配

r7.3 直接基于 r7.2，只增加可信外部 site 接线、采样 site、测试与交付说明。
没有合入 r8 的 V2 预演、V4 D2、免许可登记或正式 bootstrap RPC。
原故障机制目录 `fault_injection/`、`fault39/src/`、`native_execution_strategy/` 与 r7.2 逐字节相同。
注册不是验收：每个外部适配器和部署组合仍需当前许可、自然调用与独立产品验收。

## 安装与可信 Worker 接线

安装发行目录中三个 wheel，逐项核对 release-manifest 和 source-review。发行目录直接附带本 docs 目录和示例源码，源码包中也保留。
外部适配器可由部署方单独安装。不得让普通用户提交 Python 文件路径、模块名、裸地址或回调。
黄区继续使用自身有鉴权的可信 Worker bootstrap；每个 rank 执行以下步骤，不由 CLI 创建 Worker。

```python
from demoncat.runtime.vllm_ascend.worker_extension.external_provider import (
    ExternalSite, register_external_site, factory_sha256, ExternalFaultConfig,
)
from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import observe_deployment
from demoncat.runtime.vllm_ascend.worker_extension.product_binding import configure_product_profiles

# worker、authority、operator 来自现有可信部署；部署需审核 observed 全部字段。
# MySite 是已审核并安装的实际 Python 类，不是用户传入的 import 字符串。
spec = ExternalSite(
    mechanism="external.my_named_variant",
    factory=MySite,
    source_sha256=REVIEWED_ADAPTER_FILE_SHA256,
    expected_deployment=REVIEWED_ACTUAL_DEPLOYMENT,
    stage="model",             # 或 sampler
    region_kind="transient",  # v1 明确拒绝 persistent_weight / kv_cache
)
register_external_site(worker, spec, authority=authority, operator=operator)
configure_product_profiles(worker, profiles={
    spec.mechanism: {
        "mechanism": spec.mechanism,
        "mode": "PRODUCT_ACCEPTANCE_ONLY",
        "canary_prompts": [DEPLOYMENT_CANARY_TOKEN_IDS],
    }
}, authority=authority, operator=operator)
```

同一模型对象、完整实时部署身份和适配器文件 SHA 在使用时复核；跨 rank 适配器 SHA 必须相同。
不要只调用 factory_sha256 后自动信任任意新文件；该函数用于核对已审查的 SHA。
许可仍通过现有 install_deployment_acceptance_permit 安装，机制填外部完整名称，源码 SHA 填当前发行的 source-review.json SHA。

宿主使用原 ProductClient：

```python
client = ProductClient(
    executor, ExternalFaultConfig("external", "my_named_variant"),
    profile="external.my_named_variant", operator=operator,
    run_canary=send_deployment_canary_serially, drain=drain_real_requests,
)
client.attach()
client.capabilities()
client.enable()
# 正常 HTTP 用户请求，Worker 自动发现；宿主不传 request ID。
client.status()
client.disable()  # drain、区域恢复、资源清零、部署金丝雀
client.detach()
```

这里的外部配置只能由可信 Python 接线构造，普通 FaultConfig.from_dict 仍拒绝 external。
黄区 `python -m demoncat.cli fault ...` 的命名变体到可信会话映射需由黄区补入；本包不修改内源 CLI，也不提供任意代码加载入口。

## Site 协议

factory 接收 `(model, runner, *, fault, region_leases)`，必须保留公共框架提供的同一个 RegionLeases。
提供 `install / close / finish_forward`，以及 `events / sites / forward_calls / pending / uncovered`。
`stage=model` 在真实 compute_logits 返回后执行 finish_forward，确认所有设备消费者完成且租约为零。
`stage=sampler` 额外提供 begin_sample / finish_sample，围绕真实 runner.sample_tokens 执行；不在 logits 返回时提前收尾。

每次实际调用生成 event：site、实际 request_ids、eligible/applied/consumed/restored/failed（严格 bool）、changed（变化元素数）、
before_sha256/restored_sha256、consumer_operator。公共摘要的 changed 是变化调用数，changed_elements 是元素数。
forward_calls 按实际生成步记录调用计数；条件性调用点不做 collective。EP 不在 v1 范围。
失败租约必须保留恢复责任，不可清空列表冒充成功；未覆盖的调用写入 uncovered。

## 两个示例

1. `experiments/worker_core_v11_20261118/external_gain_example.py`：复用已审查 norm 发现与消费者观察，将真实返回区缩放 3/4。
2. `external_sampling.ExponentialNoiseSite`：用于 exponential_perturb 的采样 site。保持原 sampler 生成指数噪声 q；
   原实现完成 global stream 到消费 stream 的等待后，在真实 `probs.div_(q)` 边界将 q 的奇数词表列乘 2，
   原 division/argmax 继续执行，确认设备读取后按字节恢复 q。不改 token ID，不重写采样算法。
   固定为命名变体，不暴露用户数值/列选择参数。只接入已审核 0.26 AscendSampler 源码，其他源码拒绝。
   temperature=0 的贪心请求不注入并返回明确原因；temperature>0、单请求、非推测路径才适用。
   金丝雀使用部署固定的贪心请求，故障用户请求使用随机采样；不把随机请求间 token 不同当作故障证据。

首次适配限制：0.26 已审查同步 eager runner、单 chunk、fresh cache、单请求、最多 1024 context / 4 生成 token。
能力能否执行以 ProductClient.capabilities 与实际准入为准；旧静态目录缺项不代表外部能力不适用。

## 恢复分类与不支持项

| 区域 | 正确恢复语义 | v1 结论 |
|---|---|---|
| 临时激活 / 采样 q | 同 stream 备份完成→写入完成→真实消费者设备读完→精确恢复；中间合法更新则拒绝写回 | 支持已审查 site |
| 持久 weight_scale / int8 权重 | 所有读取者及异步副本结束，证明没有更新/重绑/派生权重缓存；持有对象与存储租约；恢复后重新验证所有使用路径 | 明确拒绝注册，缺少跨调用共享权重独占与派生缓存失效协议 |
| KV cache 写入 | 同请求后续 token 可以读取污染 cache；不能把第一个 FIA 消费完成当作整请求完成。需持有请求→slot/page 所有权，最后读完成后恢复或在干净重建边界清理；不得覆盖后续 token 合法写入或已复用页面 | 明确拒绝注册，缺少 cache 页面版本、回收与重建边界协议 |

两种持久区域统一拒绝码 `EXTERNAL_V1_PERSISTENT_WEIGHT_OR_KV_RESTORE_UNPROVEN`，许可不能绕过。
这不关闭已独立验收的内置 Native 41/26fia/20；它们有专门恢复证据，不能自动借给任意外部 site。
当前受污染请求不要求恢复为从未注入的轨迹；验收分别证明区域字节恢复、租约资源清零、新请求金丝雀精确恢复。
