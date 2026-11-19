# 0.26 A2门禁：补充证据

目标仅蓝区A2/Ascend910B4 32GiB、宿主driver25.5.0.b060/firmware7.8.0.5.201、arm64 vLLM-Ascend0.26.0rc1/CANN9.1.0。不是黄区/A3证明。冻结的旧门禁原件不覆盖。

## 流式测量

工具`tools/measure_oci_layers.py`按registry API读固定manifest，校验manifest/config、arm64/linux；每层1MiB块边读边gzip解压并累计字节，校验压缩digest/size和展开diff_id。只写小JSON收据，不写layer、tar或tensor，不调用docker pull。不重试，首个网络/解压/digest错误立即停止并保留失败计数。

固定manifest：`sha256:7265a3a08bba26b9376a4d577015ac8d459675fb3c1eec89aef6d9a19ac219d9`；17层压缩总量6,718,398,624字节。开始时间2026-11-19 13:46:25 UTC。

目前前三层通过压缩digest/size及展开diff_id校验，展开字节分别72,208,896、407,943,680、633,557,504；第4层期望4,280,972,307压缩字节，但只读到891,289,600字节即遭遇EOF；部分展开2,783,969,280字节不能作为完整层大小。**未得完整展开总量，容量门禁未通过**。本次停止，无自动重试。

实际Docker目录在独立/home文件系统；本次只读可用66,161,180,672字节。完成后重读空闲，保守计算新增压缩层+展开tar流+自有构建10GiB预算后至少留15GiB；展开tar计数不等于overlay最终分配，不能以覆盖/删除文件节省量预支空间。

## 官方配套依据与限制

- [vLLM-Ascend 0.26发布说明](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc/user_guide/release_notes.html)：0.26组合使用CANN9.1.0、torch2.10、torch_npu2.10.post4。[安装说明](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc/installation.html)要求配套driver/firmware/CANN。
- [CANN9.1.0发布说明的兼容表](https://github.com/hicann/release-management/blob/master/9.1.0/release-notes.md)：`9.1.X`行、`25.5.X`列为`Y`；推荐HDK26.0.RC1。该表支持**版本家族兼容**，未列出本机b060与精确固件的搭配。
- [昇腾官方软件共享说明](https://www.hiascend.com/zh/developer/techArticles/20260310-1)列A2 HDK25.5.0及以上/CANN8.5.0或9.1.0。这是特定功能表，不据此替代本机硬件/精确固件兼容依据，也不执行其系统配置。
- [官方驱动固件选择页](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)当前未得到本机型号、driver25.5.0.b060及firmware7.8.0.5.201的明确同表记录。driver本地version.info的firmware范围只是本机元数据，不是独立官方发布依据。

**当前结论：精确驱动/固件配套未证明，不是已证明不兼容；禁止拉取和后续安装。** 即使流式空间测量成功，也必须补足此门禁。没有改Docker、宿主驱动/CANN、0.18环境，没有清理其他数据，没有拉取0.32或做DeepSeek测试。

## 结束收据

结束时间`2026-11-19T14:16:12.976692+00:00`；状态`STOPPED_ON_FIRST_ERROR`，错误`EOFError: Compressed file ended before the end-of-stream marker was reached`。前三层合计压缩360,042,559字节、完整展开1,113,710,080字节。后13层未尝试；第4层digest未验证，不填false来混淆“不完整”和“已比较不一致”。

[原始逐层收据](../../experiments/worker_core_v11_20261118/v026_stream_measurement_20261119_night.json)，文件SHA256 `477a67fba9acac58a5e556fffaa9a68cb30f40893fe1cadcfe75f985ef469b1d`。没有层文件落盘。0.26安装、三层实装审查、39/Batch重构建和新栈NPU均未启动。
