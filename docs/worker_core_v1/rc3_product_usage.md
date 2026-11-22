# RC3 产品模式与主项目接线（开发中，未冻结）

## 定性和发行边界

r3/r4 是**机制验收工具**。`continuous_public_probe.py` 会创建临时实例，要求固定输入，不能作为黄区主项目的产品入口；A3 机制通过不等于 `product_closed`。r4 可以按原步骤继续验收，本改动不修改冻结标签和包。

产品模式复用 WorkerFaultRuntime 的 owner/generation、prepare/register/check/enable/advance/disable/unregister，以及现有 site、同 stream 消费完成和区域恢复。固定 prompt/图片仅供部署金丝雀使用。用户请求只传其真实内部 ID 到调度钩子；Worker 从本次 scheduler/requests 中读取输入，不与金丝雀逐 token 比较。

当前产品候选为 18/api_half、39/output_proxy、22/wrong_row、28/fia.denominator_replay、17/spatial_padding。候选不等于已验收；产品许可使用 `PRODUCT_ACCEPTANCE_ONLY`，旧 continuous 记录不继承为产品资格。33/split8、split16 均未通过模型传播，继续关闭，不向用户提供可执行选项。8 新代理仍在适配。其余能力分批迁移，不能仅修改 mode 就宣称可执行。

## 用户操作

以下是主项目 CLI 的契约，非运行验收脚本的命令；蓝区没有黄区主项目源码，实际 CLI 路由需由 openchamber 接线。

```text
fault list
fault enable --fault-id 18 --variant api_half --scope all
fault enable --fault-id 39 --variant output_proxy --next 3
fault status
fault disable
```

用户配置只有 `FaultConfig(fault_id, variant, implementation_version)` 和 `RequestScope(mode='all')` / `RequestScope(mode='next', count=N)`。不接收层名、rank、路径、图片 SHA、地址、原始分块数字。`next N` 按开启后进入宿主调度钩子的请求计数，超范围请求也占一个名额，并返回不注入原因；不是“挑到 N 个成功注入才停止”。模式到期停止新注入，但仍须 disable 完成金丝雀和资源清理。

## 可信部署配置示例

此文件由部署负责人维护，用户不可编辑。模型、镜像/源码及许可身份沿用部署准入；不能用此文件声明获得资格。

```json
{
  "schema": "demoncat.product.deployment.v1",
  "profile": "qwen-a3-norm-candidate",
  "fault": {"fault_id": 18, "variant": "api_half", "implementation_version": "1"},
  "model": "/deployment/models/qwen",
  "tp": 2,
  "mode": "PRODUCT_ACCEPTANCE_ONLY",
  "source": "YELLOW_DEDICATED_INSTANCE",
  "source_manifest": "/deployment/source-review.json",
  "source_manifest_sha256": "填写本次候选包source-review.json原始字节SHA",
  "native_library": null,
  "native_sha256": null,
  "canary": {"prompt": "Explain briefly why the sky appears blue.", "image": null, "max_tokens": 4},
  "evidence_directory": "/deployment/receipts/product"
}
```

`model/tp` 是现有服务身份的核对字段，不由 ProductClient 启停或重建 Worker。`source/source_manifest_sha256` 是部署许可来源。Native 三项未迁移到本产品候选，不能填库路径绕过。17 的部署金丝雀带固定 64×64 图片，预期 SHA 比对按负责人决定默认关闭，实际 SHA 仍记录；用户图片来自其推理请求。当前数值规则使用已有命名变体：28 是 block128/首块分母重放，39op 是 BF16 ND 投影单区域写 Inf；未来改变规则必须另登记命名变体。

可信宿主使用 `load_product_deployment(path)` 加载并核对文件的源码清单SHA；tokenizer和图像预处理沿用现有服务。Worker bootstrap 调用 `configure_product_deployments(worker, paths, authority=..., operator=..., canary_tokens={profile: expanded_tokens})`，再次读取文件、比对真实模型路径/TP/许可来源后安装profile。`canary_tokens`只能来自部署金丝雀的真实处理器，不能填用户prompt。这是部署 bootstrap API，不能暴露为用户传Python/任意配置的入口；它不自动安装许可。已有 `install_deployment_acceptance_permit` 继续负责许可身份。正式 Worker RPC 为 `demoncat_product_binding(action, payload)`，它必须通过已配置 authority 的 `product_binding_rpc` 授权，且仅接受预装profile名称。

## 接口与五处接线

```python
client = ProductClient.from_deployment(
    existing_executor, "/deployment/product-norm.json", operator=trusted_principal,
    run_canary=deployment_canary_runner, drain=drain_current_request,
)
client.attach()
client.enable({"mode": "all"})
# 调度钩子：普通用户请求已 enqueue，尚未执行；不把 prompt 传给 binding。
client.before_request(actual_internal_id)
# 现有推理服务继续处理请求；必须等待实际完成，而不是仅收到 enqueue 回执。
client.after_request(actual_internal_id)
client.status()
client.disable()  # 禁止新注入、drain、所有rank释放、自动金丝雀精确比较
client.detach()
```

1. **Catalog/Binding**：列出产品候选、实际登记组合、命名变体、范围及拒绝原因。旧验收记录和静态 catalog 不能显示为“产品已验收”。
2. **Controller**：映射 list/enable/status/disable；内部建立 ProductClient。部署初始化可信 RPC/profile，不接受用户部署参数。
3. **调度入口**：每个开启期间的普通请求都经过 before_request；第一版沿用串行专用实例，max_num_seqs=1。RPC 完成后才放行该请求的真实调度，不能调用验收脚本代替。
4. **结果出口**：真实完成后调用 after_request，按内部 ID 归并 rank，并用宿主已有映射关联外部 request_id。该映射必须来自实际调度器，不猜测 HTTP ID 与内部 ID 相同。
5. **关闭/清理**：停止新注入，drain 必须等待在途请求且完成结果钩子；所有 rank disable/unregister/detach，再自动运行部署金丝雀。失败时不宣称已恢复，也不解除未知状态的 owner。

客户端针对已有的**同步** `collective_rpc` executor；异步主项目须由可信宿主提供串行RPC适配，不能把coroutine对象当成完成回执。用户选择的FaultConfig映射到部署侧审核过的文件列表，不能接收用户提供的文件路径。第一次 `capabilities()` 可进行无租约的Worker身份发现；没有当前产品许可时不能由候选清单直接开启注入。现版只提供五个已接入候选的验收模式许可，不根据r4机制记录自动授予A3产品资格。

`from_deployment` 的 `run_canary(canary_config, before)` 回调必须 enqueue 文件中的固定输入，调用 before(真实内部ID列表)，再等待全部生成完成；它不接受用户输入。直接构造 ProductClient 的底层回调签名为 `run_canary(before)`。enable 要求最多六次预热内连续两次全有限且完整 hidden/logits 指纹相同，然后保留完整字节指纹基线。disable 再取同一固定输入完整 hidden/logits 指纹逐项比较。不是只比较 token、Top1 或四项统计。原始数组不在用户结果中公开。`evidence_directory`交给现有主项目结果出口保存返回值，库不会开启额外后台服务或擅自写入用户数据。

## 范围与请求结果

第一批 text 范围：真实单请求/单chunk、无 prefix cache、无LoRA/输入embedding/投机，prompt+生成不超过1024，生成1～4 token。每次调用按实际模型结构选点，全rank持续触发。17 另核对一个 uncached image 的真实 encoder 调度、处理后64×64网格和实际 embedding 消费。原始图片尺寸与处理后网格必须在部署处理器说明中区分，不凭文件名或预设图片认定用户图像有效。

超出范围时，在安装 site 之前决定不注入，普通请求仍完成，返回 `NOT_INJECTED` 与 reasons。发生了注入之后再发现映射/消费者/恢复不成立，则是故障，不能改写成正常跳过。

每个请求返回 request_id，各rank的 observed/eligible/attempted/applied/changed/consumed/restored/failed，site、变化元素数、注入区域非有限值、消费者算子及恢复摘要。没有直接观测的字段为 null，不填0。`changed=0` 与 `applied=0` 不同，任意用户数据未必产生新变化。用户请求没有 normal 基线，不报告伪造的模型级 hidden/logits 相对差异。

17 以同一真实 patch 输入执行原 padding 的局部 Native 对照，然后执行故障 padding；这个对照仅用于该算子的变化元素数，不是用户请求的模型 baseline。恢复为调用参数不泄漏及原输入/权重字节不变，不能声称曾改写输出存储再恢复。

恢复三项分别报告：每次命中区域/参数已恢复、所有rank租约/hook/owner/在途清零、关闭后金丝雀精确恢复。三项全部成立才 `restored=true`。

宿主应及时持久化 `after_request` 返回值到部署回执目录；当前客户端的status保留本作用域结果，长期无人关闭的海量请求尚未做容量验收。此版本适用已声明的串行专用实例。部分rank启动失败、实际请求未完成、恢复计数缺失或金丝雀不一致时保留失败/所有权信息，不对外报告已恢复；不能用删除容器当作同进程恢复。

## 验收进展

产品契约、正式可信RPC、部署文件加载、宿主调度/结果接口及共享site均已实现。独立审查器为 `tools/review_product_mode.py`；只读取回执，逐项重算八项计数门槛、实际消费者、恢复哈希、两组不同owner及相同Worker、完整金丝雀输出和Runtime资源计数，不根据“成功”摘要字段放行。

124 A2/0.26/Qwen3.8社区W8A8/TP2，18/39op/22/28文本版和17图像版已分别完成直接客户端与部署文件入口两作用域试验。最终无请求间缓存清理及独立next-N负向结果见[验收报告](product_mode_acceptance_20261122.md)。A3与主项目实际五处接线仍待黄区验收，`product_closed=false`；不发布或修改冻结r4，作为下一版候选审查材料。
