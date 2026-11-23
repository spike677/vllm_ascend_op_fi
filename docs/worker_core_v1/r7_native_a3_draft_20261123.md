# r7 Native A3 产品验收草稿（暂不发出）

本稿供负责人留存。当前代码只允许蓝区 Ascend910B4 / TP2 / CANN 9.1 的 Native 产品验收，不允许把设备名、来源或 TP 改为 A3 后直接运行。黄区现有构建 ff19425f / d9aa0d94 的完整提取全文、源码、编译器及收据/库身份审查通过后，才新增对应 A3 受信组合并发布执行版。没有要求黄区现在操作或重新构建。

## 库与源码

r7 保持 `fault_injection/native_adapters/postwrite` 的 C++ 原样；A2 新构建输出 SHA 为 `d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274`。它与现有记录相同不代替黄区构建审查。审查通过后直接复用黄区现有 r5 来源产物，不能用 A2 receipt 替代 A3 的 receipt，也不能借用 CANN 9.2.1 的库。若后续不得不改 C++，本稿作废并明确要求重新构建。

## 受信配置如何进入 binding

部署文件使用既有 `demoncat.product.deployment.v1`，字段 `native_library` 填 Worker 容器内只读挂载的库绝对路径，`native_sha256` 填已审查完整 SHA；`source_manifest` 和 SHA 对应本次候选源码，不能仍填 r5 的 Python 清单。各 TP 使用独立的部署配置和许可。库路径不能来自用户请求。

接线为 `configure_product_deployments → configure_product_profiles → product_binding_rpc(attach) → install_continuous_binding(native_library=(path, sha)) → NativeRegions`。宿主 RPC 只能引用受信 profile；payload 若增加库路径/裸地址字段会被拒绝。配置时和 attach 时核对实际设备、TP、库字节及部署来源，原连续 binding 再核对模型和已加载源码。候选身份不等于已登记。

```python
# 可信主项目 bootstrap。authority 必须属于该 Worker；此段不建立许可。
configure_product_deployments(
    worker, paths=[deployment_json], authority=authority, operator=trusted_operator,
    canary_tokens={profile_name: processor_produced_prompt_ids},
)
# 主项目已有许可安装逻辑随后安装与实际组合/源码相符的许可。
# 用户侧控制仅 attach / enable / status / disable / detach。
client = ProductClient.from_deployment(
    executor, deployment_json, operator=trusted_operator,
    run_canary=send_real_http_canary, drain=wait_real_requests_complete,
)
client.attach()
client.enable()
# 普通用户通过真实 HTTP 发送请求；宿主不逐请求通知 Worker。
status = client.status()
recovery = client.disable()
client.detach()
```

## 执行版验收顺序

先 TP2，再 TP1；每个 TP 独立起专用实例，eager、同步调度、无 prefix cache/MTP/KV connector，单请求单 chunk。每项 41→26fia→20 使用同一组 Worker 的两个 owner。TP1 不在蓝区预跑，也不从 TP2 自动登记。

每个 owner 先跑金丝雀至连续两次完整 hidden/logits 有限且精确一致（上限六次）。开启期间发送三条不同普通文本、每条四 token；再发五 token 范围外请求。记录所有 rank 的 observed/eligible/attempted/applied/changed/consumed/restored/failed。changed 可以小于 applied，eligible 失败必须零。没有对应真实调用的阶段保留不适用原因：41 的非 paged prefill 不算合格 KV 调用。

- 41：当前请求 ScatterPaKvCache 写槽，Native 改真实 V cache 尾列，FIA 读取对应页/槽，设备事件确认消费后恢复。
- 26fia：当前 FIA 输出复制到当前活跃 residual，add_rms_norm 真实读取，同 stream 消费完成后恢复；source/after SHA 相同，before/restored 相同。
- 20：活跃源覆盖 DynamicQuant 输入，消费完成恢复输入；真实 QMM 使用对应 Q/scale。不得改写事后输出替代输入覆盖。

关闭后验证区域字节恢复、所有 rank hooks/leases/inflight/owner 清零、新金丝雀完整 hidden/logits 精确恢复。第二 owner 不重启 Worker，不复用第一 owner。失败不能重启来掩盖残留。另做缺许可、未知 profile、用户试图传 Native 路径以及超范围请求负例。

## 回传与冻结

完整原件留黄区，采用 `review_product_mode.py` + 产品提取工具的分页文本协议；记录 Native 库 SHA、region/source/before/after/restored、真实消费者、每 rank 计数和三项恢复。执行版必须附相同源码的审查器及提取工具 SHA。A2 通过不能登记 A3。

N 批 A2 产品验收及八项产品回归通过、黄区 Native 身份审查通过后，才冻结 r7 预览；不等待 D/P。D 的 V4 噪声/全 rank 只观测和 P 的 A3 TP2/TP1 分组材料分别后续处理。
