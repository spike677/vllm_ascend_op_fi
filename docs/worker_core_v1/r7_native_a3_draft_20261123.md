# r7.1 文档补充 v2

今晚统一使用 r7.1、只做 TP2；下面历史 r7/TP1 描述仅保留溯源，不安排今晚执行。实际命令、bootstrap 和完整 Native JSON 以 [总单 v2](tp2_overnight_runbook_20261123.md) 为准。许可使用 r7.1 source-review SHA `77e30525212d2d1e8f46313452bacdd46c81ecb751ce2e052e4d72dabbf6f7e2`。Native C++/库未变，不重新构建；不修改冻结 r7/r7.1 发行包。

# r7 Native A3 产品候选验收执行单（负责人转交，A组收尾后执行）

黄区完整Native提取已通过文本身份审查，见 yellow_intake_20261123_r6_followup.md。现有库可用于候选验收，不要求重新构建。A3 TP2先行，再TP1；各组合独立许可/回执，未验收默认不登记。由负责人转交，不直接要求黄区执行。

## 库与源码

r7 保持 `fault_injection/native_adapters/postwrite` 的 C++ 原样；A2 新构建输出 SHA 为 `d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274`。它与现有记录相同不代替黄区构建审查。本次审查通过，直接复用黄区现有 r5 来源产物，不能用 A2 receipt 替代 A3 的 receipt，也不能借用 CANN 9.2.1 的库。若后续不得不改 C++，本稿作废并明确要求重新构建。

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

## r7新增运行时门禁与回传命令

可信Native配置只接受库SHA `d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274`、A3芯片原名Ascend910_9382、TP1/2及9.1.0 toolkit。attach时还读取当前Worker已映射的libascendcl.so/libruntime.so，核验ELF文件SHA及其唯一安装元数据归属9.1.0；仅版本文件或构建时cann_runtime=null不算运行时证据。无法归属时拒绝并回传，不修改准入或CANN。库仅用于Qwen0.26/CANN9.1，DeepSeek0.32/CANN9.2.1不可混用。

以下使用黄区CLI实际产品回执和部署身份，TP2先行；三机制各两个owner，同组Worker，不用蓝区实验脚本代替黄区CLI：

```bash
# TP2：分别对41、26fia、20实际保存的回执执行；TP1重复，--tp 1。
python tools/review_product_mode.py "$RECEIPT" --tp 2 --a3-native --output "$REVIEW"
python tools/extract_product_record.py "$RECEIPT" --runtime-identity "$RUNTIME_IDENTITY" > "$TEXT"
```

runtime-identity文件按工具DEPLOYMENT字段填写实际值，缺失填null；源代码清单SHA使用r7许可清单。提取页包含运行时ACL身份、库SHA、计数、消费者和三项恢复；≤4096字节硬上限，原样分页回传。额外提交缺许可、未知profile、用户库路径注入拒绝结果及对应owner/rank；审查器未通过须保留失败文本。原件留黄区，构建提取的反斜线转义修复只限已记录的传输解析，运行回执不修改。

安装r7到独立环境；先按交付清单核验三wheel、源码包、source-review和安装源码，绝不覆盖r6。A3新组合仅候选，product_closed仍为false。r7已同步r6.1串行B1金丝雀修正；旧TP1 Batch失败原因未唯一归因。新协议A2 TP2通过，黄区TP1/TP2仍分别复测。


正式发行使用标签 demoncatpfi-rc3-preview-20261123-r7；完整包SHA和许可source-review SHA以 releases/RC3-PREVIEW-R7-HANDOFF-20261123.md 为准。本文虽保留历史draft文件名，执行内容已定稿；必须与r7同包使用。
