# r8.1 登记骨架与免许可冒烟（黄区按D73自验证、自冻结）

新D73替换此前蓝区复审前置：黄区负责人/黄区codex自行审查、登记、冻结与合入，蓝区只处理升级单。本地蓝区维护分支 `codex/rc3-r8-1-registration-20261126` 起点 `e38c4113b421ac979699071a60e80d3babd05e07`，当前零改动。最终差异只允许 `src/demoncat/runtime/vllm_ascend/worker_extension/product_registered_records.json`。本执行说明在开发文档中，不塞进只改登记表的维护提交；r8.2 的 Batch/诊断修复独立。

候选为 A3 / Qwen3.8 W8A8 / 0.26 / CANN9.1.0 / TP2 的 11 项：18.api_half、18.variance_over4、22.wrong_row、15.alias_detach、39.output_proxy、39.matmul_range、28.fia.denominator_replay、17.spatial_padding、41.tail_column、26.fia_destination、20.buffer_substitute。候选不是放行。外部三项、Batch、under4 不在本次公共登记范围。

## 结构（仅示意，不能载入注册表）

```python
from demoncat.runtime.vllm_ascend.worker_extension.product_registration import (
    IDENTITY_FIELDS, validate_identity, validate_record, digest)

# rank0_identity / rank1_identity 必须来自黄区原样回传的实际 observe_identity。
# 不用默认值填充；当前七项双rank身份已入独立草稿，负例未齐，不构造可执行记录。
record = {
    "schema": "dcat.product.registration.v1",
    "source": "YELLOW_DEDICATED_INSTANCE",
    "mechanism": mechanism,
    "expected_ranks": [rank0_identity, rank1_identity],
    "evidence": {
        "mode": "product", "level": "YELLOW_TEXT_ATTESTED",
        "review_passed": True, "owners": [owner0, owner1],
        "receipt_sha256": reviewed_original_sha,
        "reviewer_sha256": reviewed_reviewer_sha,
        "source_manifest_sha256": reviewed_source_sha,
        "consumers": True, "bytes_restored": True,
        "resources_zero": True, "canary_recovered": True,
        "negative_controls": True,
        "coverage": reviewed_owner_rank_counts,
    },
}
# coverage 为两 owner × 两 rank，每项含 owner、rank、counts 八项。
record["record_sha256"] = digest(record)
validate_record(record)
```

每 rank 身份键必须与 IDENTITY_FIELDS 完全一致，逐项 validate_identity。rank/逻辑设备允许不同，其余跨 rank 公共身份必须一致。Native 需另外验证构建元数据、库 SHA 和 CANN 映射。effect 页的部署子集不能替代这些字段。原文缺件时不写任何占位记录到加载路径，现有 BLUE 记录原样保留。

## 发布后免许可冒烟

沿用 r8 的四槽分组，A系列物理卡用 8–15：A0=8/9、A1=10/11、A2=12/13、A3=14/15。本轮A2并入A1、不占12/13；D/G已在0–7的v4b-07/19301独立运行，不再与A系列错峰或互相等待。冻结包不就地改写。r8.1登记与冻结已由新D73授权黄区自行完成，详见仓库根目录 `releases/YELLOW-HANDOFF-20261126.md`；公共代码和旧冻结包仍不改。

1. 独立安装 r8.1，核对 wheel、安装源码与注册表 SHA。停止旧 API server，确认健康检查连到本次新进程；保留相同模型、Native 库、processor、参数范围。只有通过登记的精确组合可使用以下步骤。
2. 所有 rank 清除旧验收许可，确认许可不存在、无旧 owner/租约/在途请求。由可信部署的 bootstrap 配置原机制 profile，唯一模式变化为 `PRODUCT_REGISTERED`，不再 install_deployment_acceptance_permit。保持部署鉴权。
3. `ProductClient.capabilities()` 必须显示 PRODUCT_REGISTERED、product_qualified=true 和相同 record SHA；任一 rank 不匹配即拒绝，不降级到验收许可。
4. 每个已登记项按原可信 CLI/session 接线做 baseline → enable → 用户请求 → status → disable（金丝雀）→ off → detach。参数 temperature=0、seed42、logprobs5、max_tokens4。17 仍用真实图像；Native 仍用已登记库。保存实际请求、每 rank 覆盖/消费者/三项恢复及 permit 不存在的证据。每项一次冒烟，39op/41/26fia 两个 owner 回归按 r8 执行单要求保持。
5. CLI 命令仍为黄区 `python -m demoncat.cli fault list/enable/status/disable --socket "$SOCKET"`，enable 另传 `--fault-id "$MECHANISM"`；可信会话由宿主建立，CLI 不创建 socket。disable 后确认完成 detach 和资源清零。
6. 负例：未登记的组合返回 PRODUCT_COMBINATION_NOT_REGISTERED；身份漂移、跨 TP 借用、参数范围变化、缺必需字段分别结构化拒绝。CLI exit4 和拒绝码，引擎存活，资源零，随后原精确组合正对照。不得改生产模型文件制造负例，使用隔离测试配置或 CPU 身份夹具。
7. 保存标签、源码、注册表/record SHA、两 rank 身份 SHA、owner/原件对照、免许可正例和负例。由黄区结合免许可冒烟、负例和master验证自行决定product_closed，并记Y-D决定及里程碑；不等待蓝区复审。r8.1发布本身不等于产品闭环。

证据未齐前本单只是准备稿；不得把示意 True 或未知 SHA 当实测值。

11月26日17时补充：22/15的新页面与产品提取通过；over4仅效果页面通过，产品路径和提取完整性均未通过。完整身份已收到rank0文本/视觉/Native三类，但rank1仍缺；Native构建元数据为空，负例尚无有效结果。详见 `yellow_r8_followup_review_20261126.md`。本维护分支继续零改动，不发布。

随后收到rank1文本/视觉/Native身份，覆盖上述历史缺件中的rank1。七项双rank草稿已写入 `r8_1_expected_ranks_draft_20261126.json`，14份身份通过校验，negative_controls仍为null。Native sidecar已生成但黄区实际观测待回传；当前结论以 `yellow_r8_identity_owner_review_20261126.md` 为准。运行时登记表仍不改。


## D73当前执行权与安装身份

上述历史缺件记录保留，不是要求等待蓝区。黄区证据齐后validate_record自检，做只改product_registered_records.json的提交并自行冻结；保留旧SITE，从该SOURCE构建独立新SITE。逐条记录SHA、安装检查和免许可正负例由黄区负责，蓝区不常规复审。

implementation_identity只散列worker_extension和fault_injection的.py；登记JSON不参与。安装后实算必须仍为`003294ac9e7468b2069719a44f9931514a9ce504c6b12e6c39f64684878f0bee`，同时核对实际导入路径。不能安装交接分支的C2 Host候选然后沿用此身份。超权限问题按交接单≤4KB升级。
