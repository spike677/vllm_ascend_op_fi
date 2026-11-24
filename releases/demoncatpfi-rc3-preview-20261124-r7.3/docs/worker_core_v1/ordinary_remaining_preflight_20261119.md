# 剩余普通任务：实施前适用性预判

本表是 RC2 冻结后的开发判断，不继承历史证据为当前普通入口资格。124 可见两张 Ascend910B4、每卡32GiB；每次仅占一张空闲卡，使用资源锁和实时门禁。2026-11-19本次只读检查卡2/5均无计算进程，但这不代替每次的设备预留检查。

| 能力 | 已有证据及部署 | TP1串行eager候选 | 本轮需要解决的阻塞点 |
|---|---|---|---|
|20 buffer_substitute|QMM自然推理的原件审查见 agent7_native20_independent_review、serving20_qmm；历史报告不完整模型身份项保留为null|本地Qwen3-0.6B W8A8；须在新作用域重新确认真实QMM及Native映射|旧provider可能回落research replay；report会collect，须隔离普通路由并改为真实请求回调完成、query只读；不能删去R/L/OFF资格检查|
|26 residual_stomp|Source54 ce82ca5f 单QMM，后续Source68实际TP1；135调用里2稳定、106已证不相邻、27未解决|本地量化小模型TP1存在历史单点条件，不因早期32B TP2记录判定不适用|每次重验真实allocator相邻独立块，residual末端等于output起点、padding+32≤4096；不能搬移/伪造相邻，不把2个稳定点说成完整分母|
|26 fia_destination|Source62 dc40a87d typed-FIA自然prefill单点，30实际调用；原始数据路径见 fia26_source62_review.json|历史小模型推理路线可作为候选；不是训练注入|当前Native typed描述符、实际writer/residual/consumer、物理邻接和全量恢复须重新成立；旧保存的通过标记不是入场凭证|
|33 paged_decode|Source105 8796f84d、Qwen3-0.6B FP16 TP1 TND首decode，Q1/KV1025/Hq16/Hkv8/D128/page16|本地小模型资源可满足；只沿Source105|新源码/包的stock/private完整normal桥接仍必须重做/审核；不能复用旧435对数据授权新包。只一个有proof的实际目标，另27个条件匹配调用保持未验收|
|41 tail_column|Source60标准Worker：Qwen3-0.6B BF16 TP1串行eager，真实writer→live KV slot→TND FIA；见 core41_source60_review.json|本地BF16小模型具备历史自然调用链|17请求研究协议需拆出普通单点任务；保留两次健康观察与lossless、实际cache恢复；回调自动结束请求、query纯读、独立health及两个owner资源清零|

历史入口和机制不删除。新普通入口实机通过前默认关闭；CPU测试不授NPU资格。单点选择不能授全目标C2/C3。两个独立作用域需各自确认实际自动选点、Native命中、效果、OFF和health精确恢复、资源归零。

同一具体原因连续失败两次即停止该项，保留两次原件、失败码和作用域，转下一项；不重试F以替换失败记录。当前是适配预判，未声称五项已重新实测。若新部署缺少机制前提，记“当前部署不适用”，不能以新机制替代。

28只继续真实TND denominator路线，F/OFF未闭合前维持关闭；不新建legal变体或退回HF入口。RC2发行包不包含这些新开发结果；后续通过项另发RC2.x。RC2.1专供黄区登记，不混入这里的机制修改。

## 0.26 A2门禁（阶段D）

[本次原始元数据及判定](../../experiments/worker_core_v11_20261118/v026_prepull_gate_20261119.json)：实际Docker目录`/home/docker-system`位于独立`/home`文件系统，约61.64GiB可用。arm64镜像压缩层6718398624字节，镜像配置指向CANN9.1.0；registry manifest/config/Quay标签元数据均未给出解压总量。宿主驱动25.5.0.b060、firmware7.8.0.5.201，尚未获得对该精确A2组合的官方兼容依据。

结论：**停止在拉取之前**，解压后容量及驱动兼容性未通过；不是已证明不兼容。不用压缩比猜解压空间，也不以他人的CANN9.1容器能运行代替兼容证明。没有拉取镜像、改驱动/CANN/Docker、清理其他数据或动0.18。0.26无设备安装和后续重构建/两作用域实测尚未开始。

公开核对入口：[v0.26 Dockerfile](https://github.com/vllm-project/vllm-ascend/blob/v0.26.0rc1/Dockerfile)、[官方驱动配套查询](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)。动态页面当前默认选项不是本宿主兼容证明。
