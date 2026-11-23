# RC3 预览 r5.1：仅登记维护发行

从r5发行提交`a76f33cfbe97dbbf57955029dc3225d98985de6a`建立独立维护分支`codex/rc3-r5-1-registration-20261123`。源码提交`6bc79f824607cf4a3bfac16c9021aaef0bce5cd6`，标签`demoncatpfi-rc3-preview-20261123-r5.1`。唯一运行时源码改动是`reviewed_deployments.py`登记数据；没有合入r6机制、产品代码或新提取工具。

## 登记范围

14个`YELLOW_DEDICATED_INSTANCE`：A3 / Ascend910_9382 / vLLM0.26.0、vllm-ascend0.26.0rc1、torch2.10.0+cpu、torch_npu2.10.0.post4、CANN9.1.0 / 黄区现役Qwen3.8 W8A8 / TP1与TP2各七项。config、量化描述、全部权重索引SHA、部署参数及设备映射分别保留在登记表，未补历史默认值。

|机制|TP1 registration_ready|TP2 registration_ready|
|---|---|---|
|28 denominator_replay|true|true|
|Batch shared_scale|true|true|
|18 api_half|true|true|
|22 wrong_row|true|true|
|39 output_proxy|true|true|
|39 matmul_range|true|true|
|17 spatial_padding|true|true|

每项两个独立owner、八项计数、消费者、恢复、负向检查及r5许可身份均通过文本复核。证据等级`YELLOW_TEXT_EXTRACTED_ATTESTATION`（黄区文本提取证明），完整summary与原回执保留黄区，蓝区没有审查原件。原提取文本的null与extract_sha256未修改，逐份结果在发行目录`registration-review.json`。

image_digest按负责人方案B记为“未观测（4.9 非必需）”，不进入expected身份比对。标签v0.26.0rc1-a3及本地Id `sha256:1a9d274a0f509583fb8b020aad8c5ba53fa045e5162ecb3146c18452d30b6b8d`仅为黄区补充备注，不称为回执观测。

changed小于applied不是失败；八计数的应用/消费/恢复守恒及failed=0才是对应门槛。22 TP2两作用域约1.35/6.32倍、39mr TP2约23.7/9.18倍仅记录，不作性能结论或门禁。

## 准入边界

登记放在`YELLOW_CONTINUOUS_RECORDS`，与历史ordinary/Native的`YELLOW_RECORDS`分开。冻结r5的持续入口仍要求原有可信作用域许可；本维护版不修改该准入算法，也不将连续验收记录错误授予旧Native/ordinary路径。登记是上述14个历史验收组合的审查结论，**不是产品模式资格**；product_closed=false。17的历史固定图结果有效，不在r5.1改图。33、41、26fia、20、8和DeepSeek没有新增资格。

## 包、安装及核验

[发行目录](demoncatpfi-rc3-preview-20261123-r5.1/)包括三wheel、源码tar、源码清单、发行清单、校验清单、登记审查、CPU检查与124无设备安装回执。许可字段source_manifest_sha256对应**source-review.json原始字节**：

```text
b140ee4b091f5291143ba072e8d135f21eadd801762e448a0b0fe4924def03cf
```

历史证据的许可SHA仍是r5的eff3ec54…，不重写为r5.1许可。三wheel保留r5版本0.4.0rc4/0.3.0rc4，必须按本发行SHA装到**新的空目录**，不能让pip按相同版本号复用旧安装。原r5/r6标签和包均未改动。

```bash
cd "$RELEASE"
sha256sum -c SHA256SUMS
mkdir "$SOURCE" "$SITE"  # 两者都必须是新目录
tar -xzf demoncatpfi-rc3-preview-20261123-r5.1.tar.gz -C "$SOURCE"
python -m pip install --no-index --no-deps --target "$SITE" "$RELEASE"/wheels/*.whl
python "$SOURCE/tools/check_rc3_preview_install.py" --site "$SITE" --release "$RELEASE"
```

48项与登记/准入相关CPU测试通过；源码tar1122文件逐字节核对通过；124 A2隔离容器、无设备、无网络安装检查378文件一致（2026-11-23 04:00:33–04:00:37 UTC）。该安装检查不加载模型、不创建Worker、不初始化NPU，不是A3实测。

本说明由负责人转交；没有向黄区发送消息或修改黄区master。后续产品A组使用独立r6环境及新A组执行单，不把r5.1验收登记当作r6产品通过。
