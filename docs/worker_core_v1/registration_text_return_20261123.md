# r5登记与DeepSeek紧凑文本回传

本工具只读本地文件、向stdout打印，不发网络请求、不写许可或登记表。不要求传原件/summary文件或全文。原件留黄区。每个r5组合单独运行，最多14段，各段完整粘贴，不手动编辑JSON。目标≤2048字节，硬上限4096；超硬限拒绝输出，不截断索引或证据。超过2KB而未超过4KB在stderr提示。

在工具包根目录运行（Python标准库即可）：

```bash
python -B -m tools.extract_registration_record /path/summary.json /path/continuous_public.json --release-manifest /path/r5/release-manifest.json
```

`--release-manifest`可选：仅当该文件source_manifest_sha256与回执**各rank实际使用的许可**一致，才提取源码提交和包名。缺失字段输出null，不从镜像标签、历史脚本或蓝区补值。镜像digest必须存在于运行记录的actual.image_digest；没有就保留null，登记前须补只读观测证明。不能把本次提取工具的源码SHA当作r5包源码SHA。

工具在黄区重新运行附带的review_continuous_public.py，启用require_negative_creation；核对summary每个原审查字段与重跑结果，核对source_receipt_sha256。失败时review第一位false、第二位错误原因，仍可回传；不会自动登记。

紧凑字段（顺序固定，schema=dcat.reg.extract.v1）：

- p=[标签,源码提交,许可source_manifest_sha256]。r5许可SHA为eff3ec54d292deec219555f32fc928278c30c23541e3ef794ef7f89f152a2fff，对应source-review.json。
- d依次为hardware_family、hardware_device_name、vllm、vllm_ascend、torch、torch_npu、cann、image_digest、model_config_sha256、quant_model_description_sha256、全部safetensors_index_sha256字典、max_model_len、max_num_batched_tokens、max_num_seqs、tp、dp、pp、eager、dtype、live_config_matches_file。各rank/作用域不一致的项为null并列入conflicts，不能折叠成一致。
- devices为去重的[rank,逻辑设备]列表，不输出机器地址/容器名/硬件序列号。
- m为机制；s每项=[owner,rank列表,两负向结果,耗时倍数]。
- rank行=[rank,八计数,消费者结论,[区域字节恢复,资源清零,OFF且health精确]]。八计数依次observed/eligible/attempted/applied/changed/consumed/restored/failed。负向依次多余advance拒绝、缺许可拒绝创建且清理。
- review=[本地复核通过,错误,summary与复核一致,summary链接原件SHA一致,审查器源码SHA]。
- files按summary、原回执顺序列[文件名,字节数,SHA256]。
- extract_sha256：移除此字段后，JSON sort_keys=True、ensure_ascii=False、separators=(',',':')、allow_nan=False，UTF-8无BOM无换行，SHA256。传输末尾换行不计入。

蓝区离线检查（只检查提取证明，不宣称持有黄区原件）：

```bash
python -B -m tools.extract_registration_record --verify saved-extract.json
```

最终仍要按r5身份范围核对现役量化描述及全部权重索引SHA、镜像digest、安装检查。提取hash证明传输一致，不是原件独立可验证的数字签名。证据等级为“黄区文本提取证明”；r5.1仅登记，不改机制。当前未收到14段提取文本，不能提前发行r5.1。

## DeepSeek噪声和只观测

新标定工具在成功完成有限值和token稳定校验后，summary明确输出all_finite/tokens_stable；旧summary缺这些字段仍为null，不推定通过。私有完整数组不回传。

```bash
python -B -m tools.extract_dsv4_record noise /path/noise-summary.json
python -B -m tools.extract_dsv4_record observe /path/observe.json
python -B -m tools.extract_dsv4_record observe /path/source-fingerprints.json
```

每行独立UTF-8 JSON≤4096字节，包含原文件SHA、page/pages、提取SHA，所有页均需回传。噪声每rank一页，保留4个step各项maxima/bounds、1.5系数、token和prompt身份、两进程样本数；顺序hidden_max_abs/logits_max_abs/top5_rank_logprob_max_abs/top5_set_distance。只观测按rank/机制/site分页，保留调用路径、shape/dtype、阶段、计数和缺失原因；请求ID只输出数量+哈希，源码每文件一页，缺失SHA保留null。不把候选eligible等同实际可注入。任何页超限直接报错，不能删字段凑大小。

以上内容交负责人转发，本轮未向黄区发送任何消息。
