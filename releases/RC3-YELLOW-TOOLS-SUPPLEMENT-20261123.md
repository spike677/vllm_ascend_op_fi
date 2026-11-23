# 黄区工具补充件 r1（2026-11-23）

独立交付，不修改r5、r6标签或发行包，不并入r5.1。补齐之前发行遗漏的DeepSeek采集/只观测脚本，以及Native现有构建提取；同时附方案B登记verify、产品TP1/TP2文本复核和r6 A组执行单。该工具包不授权故障，不登记产品资格。

- [ZIP](yellow-tools-20261123-r1/yellow-tools.zip)：80524字节，SHA256 `dd46f30f9c268758b4a70fd78d6134c3d74e453cb801ef1de7c224bd44e6f0c1`。
- [逐文件manifest](yellow-tools-20261123-r1/manifest.json)：39个工具/依赖/文档/测试文件，含字节数与SHA；ZIP内部另含manifest及SHA256SUMS。
- [外围SHA256SUMS](yellow-tools-20261123-r1/SHA256SUMS)、[独立解压CPU检查](yellow-tools-20261123-r1/cpu-check.json)、[CPU原始日志](yellow-tools-20261123-r1/cpu-check.log)。
- Python≥3.10。纯提取/审查只依赖标准库；capture/observe用专用镜像现有torch/vLLM/torch_npu。无需下载权重、不安装软件栈、不构建Native、不创建Worker。

|文件|SHA256|
|---|---|
|`tools/extract_native_build_record.py`|`8a4a5352a9ad2f5d1c2f5a06d0911829f61bb596258f560c810d99cadfb48036`|
|`experiments/worker_core_v11_20261118/dsv4_noise_capture.py`|`925f7604bba677ccb716bd9e84b0eaa429951010f3ec0720901e47812f2fc6b8`|
|`experiments/worker_core_v11_20261118/dsv4_observe_probe.py`|`4bf95cdc05626b6101aa030b3c4fa6397857bc08430e9ecc57b778b188c28c57`|
|`tools/dsv4_capture_bundle.py`|`a7b07bc2fcdfb8acdd920122cc5cd38623e026bd8b666f037537f65aba3d95ca`|
|`tools/extract_dsv4_record.py`|`33e9fdad80120e522606be098340ae163ebef17b13db10650a6d097f60b32b50`|
|`tools/extract_registration_record.py`|`f648e3ff1e9361bf9e23e77852715d647268e89fc1a2bac7b5d91be9df3ca2a2`|
|`tools/review_continuous_public.py`|`deb3c5c0d600f7bcae7d0997cfd883591ece87231ed8128e486edb5d823ffadc`|
|`tools/review_dsv4_noise.py`|`e595e3013e544f9350e86e8b00acafd862e08f55740e8865b98404a0f68b182b`|
|`tools/extract_product_record.py`|`5d9ed35256c15d406d6fd8370811efd94b70ab3c308a1ecd6dbf22c5ae705c83`|
|`tools/review_product_mode.py`|`605dfce013a0db2e346277c12a45bcba66bc5e9b0a3dc1799bd65882f0e34c7d`|

r5审查器SHA与原r5归档逐字节相同。新版登记verify仅将image_digest列为非必需，其余身份缺失仍拦截。原14段提取文本不改、不重新计算extract_sha256；独立解压包实际运行14次CLI verify全部registration_ready=true，再运行一次逐字节相同。此包不包含冻结r5源码tar；整份发行集成/图像测试在开发树执行，独立包只跑自身依赖范围。

## 操作步骤（由负责人转交）

解压到全新TOOLS目录，不覆盖r5/r6/现役模块。先在包目录运行`sha256sum -c SHA256SUMS`。逐条命令、参数来源、实例及回传格式见[完整工具说明](../docs/worker_core_v1/yellow_tools_supplement_20261123.md)，ZIP中带同一文档。

1. **Native**：对现有receipt.json与libdcat_postwrite.so运行`python -m tools.extract_native_build_record ... --deployment-identity ...`，不重建。runtime-identity填写已知完整1a9d274a…镜像Id、repo_digests=[]、cann_runtime=null（无设备构建只有SDK文件版本），实际源码发行/提交未知则null。工具打印source_files、compiler、SDK、每个命令返回码、实际库/回执SHA与缺项，不静默补值。
2. **噪声**：DeepSeek V4-Flash-0731 W8A8、A3、v0.32.0rc1-a3、TP8+EP，先核实实际加载CANN9.2.1。用capture收集真实4步；assemble_run保留真实请求/进程身份；两个独立Worker进程组各≥5样本；review_dsv4_noise产出私有summary，再extract_dsv4_record noise只打印8rank文本页。不是W4A8诊断，也不是用默认身份补齐。
3. **只观测**：同一DeepSeek部署，observe source读取实际已加载包的源指纹，start/stop覆盖真实推理。结果用extract_dsv4_record observe分页。unknown、缺失、observer移除情况全部保留，不把0当作不存在。
4. **回传**：每页原样粘贴纯文本，≤4096字节；完整summary、hidden/logits、原回执留黄区，仅回传其字节数和SHA。所有工具在输出前检查上限，超出拒绝，不截断证据。缺少实际采集接口时停止并报告；CPU成功不代表0.32实机已经通过。

## Native审查仍未完成

ff19425f文本已给出SDK9.1.0、库SHA及9057字节receipt身份、构建/CPU命令成功；还缺**实际构建源码发行/提交、各源文件SHA、编译器版本与提取输出**。详见[审查清单](../docs/worker_core_v1/native_build_text_return_20261123.md)。收到后逐项核对既有发行源码；在此之前41、26fia、20产品模式保持关闭。无需为了回传字段重新构建。

## r6 A组现在可独立执行

[A组执行单](../docs/worker_core_v1/r6_yellow_group_a_20261123.md)含独立安装、CLI到ProductClient公开接口、TP1/TP2各七项、两个owner、真实HTTP并发Batch与B1负例、内置17金丝雀图与spawn、提取回传。33只做A3 launch观测及固定QKV控制≥99%，仍关闭注入，不能沿用A2分组。当前33取证probe固定TP2，TP1需要独立适配审查，明确记为待办，不伪造通过。

Qwen模型1的Native构建/验收固定v0.26.0rc1-a3/CANN9.1.0；DeepSeek模型2固定v0.32.0rc1-a3/CANN9.2.1。Native库不得混用，在另一CANN构建属于新组合。该约定补充r6及以后交付说明，不改冻结包。

## 验证

开发树相关CPU测试62项通过；独立ZIP解压后43项通过，包含真实CPU张量调用的capture→review→extract及observe→extract、Native CLI重复提取/损坏库拒绝、TP1产品回执及故障金丝雀拒绝。另14份真实黄区提取文本在独立包内通过stdlib-only CLI verify，两次输出一致。所有CPU数据明确为夹具，不代替DeepSeek/A3验收。没有新NPU调用，没有卡占用、抢占或隔离。
