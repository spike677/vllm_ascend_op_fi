# 附件799977执行与交付记录

范围：10个故障库行的黄区参考工厂，A2 / Ascend910B4 / vLLM 0.26.0 / vllm-ascend 0.26.0rc1 / CANN9.1 / 社区Qwen3.8 W8A8 / TP2。全部为**现象复现，非原机理**，不是新的公共机制，不授予A3资格。

## C1与冻结边界

K文档和正式测试分阶段交接已提交；`releases/qwen-r741-phased-handoff-20261124/handoff.zip` 的SHA256是 `fb723d5cbc559151940b700cd991dd2d691c47d5827792568d3f65f669c89d4f`。33补充r2与N2说明通过该交接单索引。r7.5文档同步提交 `a347dddf` 已推送。

本轮新工厂只放 `reference_adapters/`。没有修改 `src/` 或 `fault_injection/` 的共享产品实现。冻结r7.4.1标签、源码包、wheel及许可保持原样。依赖的Git内容与冻结基线相同；Windows工作树部分CRLF与冻结源码包LF的字节差异在补充包 `dependency-byte-forms.json` 中逐项披露，不声称两种字节相等。

## C2证据与交付

逐项状态由 [41行JSON](fault_library_coverage_20261124.json) 生成 [台账](fault_library_coverage_20261124.md)，不手写进度合计。完整紧凑复核记录在 [A2结果JSON](reference_adapter_a2_20261124.json)，含原回执及侧录SHA、源提交/清单SHA、实际部署身份、每rank两个owner、同组Worker身份和最终恢复。

每个独立补充件含factory、逐文件SHA、冻结依赖SHA、固定配方、命令、A2复核记录和交接说明。仅manifest列出的通过变体可供黄区审查接入；同一源文件内未列出的类不因此获得资格。正式测试与登记仍由黄区在其部署身份上完成。

|补充件|库行|ZIP SHA256|
|---|---|---|
|`releases/yellow-reference-swiglu-20261124-r1`|3|`22de08405a52228d9f4771ec32f75bf65b10bebc2f1b68a506215cc54f36d6fe`|
|`releases/yellow-reference-quant-20261124-r1`|9、10，各两档|`dbcb1da1346f02d39ca921dfa78c517fece9f179d9704da937ef14f5aaf61cf3`|
|`releases/yellow-reference-hidden-20261124-r1`|7、8|`c3aed8dd84bb533adc45276cff55882c81b09b628e9bde14a0eaec5aa592b246`|
|`releases/yellow-reference-vision-compute-20261125-r1`|16|`83145839a481d1a111ebb78b62c566e421f61ea2a459cb03a0d8d69d348d6da3`|
|`releases/yellow-reference-matmul-order-20261125-r1`|19|`579d6838fe10973322df80f9c227ebb00156aa4e3bfe00efa5012bf3da5de76e`|

以上ZIP逐文件bytes/SHA及归档SHA均重新校验通过。

以上均已在独立目录以冻结r7.4.1源码加补充ZIP解压，执行对应CPU测试。实机结论来自原始回执的独立复核，CPU测试不替代实机。

## 失败与修正保留

- 量化初始两次尝试在注入前被测试工具错误拦截：外部机制错误使用内部FaultConfig、随后错误预期内部缺许可码。已仅修测试工具；正确外部拒绝码为 `EXTERNAL_CURRENT_ACCEPTANCE_PERMIT_REQUIRED`。
- 三档合并测试的一次资源预算超过单case1800秒门禁，在NPU启动前拒绝；随后使用满足预算的同Worker三档配方。
- 第7、8行第一轮虽计数一致，但消费者是 `vllm::maybe_chunk_residual`，实际只读x的形状。独立源码审查否决该结果；工厂保持租约直到 `_C_ascend::npu_add_rms_norm_bias` 真正读取，修正版重新实机通过。审查器新增对形状-only假消费者的拒绝测试。
- 第12行前两轮未观察到softmax，测试引擎退出，未计为通过。实际vLLM采样器会先把BF16 logits转FP32；修正版追踪真实转换后的存储来源，保留合法温度写入，再在真实softmax短窗口内租用和恢复。第二轮进一步查到lmhead TP路径在FP32转换前取 `logits[:input_batch.num_reqs]`，产生同存储的新视图。仅允许实际首请求完整词表行的连续视图，拒绝同值副本和错行。第三轮实际视图修正版仍报相同错误，0个owner完成，引擎退出；停止同原因NPU重试，不交付为通过项。未证明真实softmax消费者，也不能用CPU测试替代。
- 设备门禁一度将Kubernetes释放后的显式 `PodDeviceEntries=null` 误判为未知格式。独立测试控制器兼容显式空切片；缺字段和其他未知格式仍拒绝。宿主checkpoint、Docker配置和产品准入均未改。

- 第32行实机两个owner、每rank六次真实Conv3d/encoder消费及恢复通过，但BF16图像输入的FP16往返changed=0。结论为 `EXECUTION_RECOVERY_PASSED_NO_NUMERIC_PHENOMENON`，不加入通过补充件，不加额外扰动制造数值变化。
- 第19行首轮真实BF16矩阵乘→all_gather，每rank24次应用和恢复，前64列changed=0。后续固定档观测前1024列，并分别记录control_changed_elements（原生对正序参考）与rule_changed_elements（正序对逆序）；仅前者变化不能证明累加顺序效应。

- 第34行两个rank均报 `REFERENCE_NO_REAL_BF16_WEIGHT`，引擎随后退出。已确认社区视觉矩阵权重为int8，当前自然BF16带bias点位不适用。未完成两个owner，不能称为安全拒绝或产品通过；不解量化权重、不添加假bias/residual。失败日志SHA保留在A2结果JSON。

- 第19行前1024列复核通过，但效果必须分rank披露：每rank24次，rank0 changed=2/control=2/rule=0，rank1 changed=4/control=6/rule=8。独立审查按每owner的Hamming计数下界 `max(0,rule-control)` 证明至少1元素的顺序效应；两owner合计下界2，仅rank1。初版复核器额外要求每rank均有顺序差异，已改为每owner必须有可归因效果、所有rank应用/消费/恢复完整；这遵循changed<applied不是失败的口径，不把rank0控制漂移算作顺序效应。原始回执和侧录没有修改，独立审查器SHA另存。

## 资源窗口

本轮使用项目锁、每case预算和有界只读设备监视。没有停止外部容器，没有修改调度器或Docker入口。窗口结束停止监视线程并释放锁；记录中的 `isolation_removed` 表示本项目监视已结束，不表示修改过外部调度器。

- `refquant1124a3`：0.5档，318.074秒，窗口已结束。
- `refquant1124a6`：0.25/2/4三档，同Worker，480.109秒，窗口已结束。
- `refhidden1124b3`：7/8修正版，同Worker，304.123秒，窗口已结束。
- `refs1124b1`：含旧7/8、第12行失败及第3行通过；总窗口1059.537秒。中途空分配解析拒绝后已退出；旧7/8不能作为通过证据。

- `refs1124b4`：UTC 15:58:15.140706 → 16:24:13.450883，1558.310秒；12失败、16通过、32/19无数值现象、34无合格候选。只读监视已结束、卡锁已释放。

- `refmatmul1125c2`：334.035秒，两个owner完成，监视已结束、卡锁已释放。

- `reflogits1125c1`：250.503秒，消费链未闭合；监视已结束、卡锁已释放。最终核对三个窗口均已结束，没有本轮残留测试容器；未停止外部容器或修改调度器。绝对时间和文件SHA见A2 JSON的final_window_audit。

这些是验收窗口墙钟时间，包含加载、预热、HTTP和取证开销，不冒充纯NPU计算时间。`unregister` 中间状态尚未执行最终金丝雀；恢复结论读取 `disable.recovery` 和独立审查的 `canary_exact/all_resources_zero`。

## C3、C4、C5

33等待黄区N2六条目标记录补齐关联和字段，收到后才能锁定serializer和分组、做三形状×两rank的≥99%检查点。没有把2488字节候选结构强套到不匹配实测。

D包等待D0/D1及噪声审查输入，仅覆盖公共8与库13。没有用V2-Lite预演代替V4资格。

截至本轮核查，接收目录未出现新的N2、D0/D1或正式产品提取。参考项通过只改为 `ADAPTER_READY_UNTESTED`；黄区正式回传逐份审查后才能 `REGISTERED`。r8整合和免许可登记依赖这些结果，未提前放行。所有交接材料由负责人转交，Codex不直接向黄区发送。

## 本轮交付边界与未完成项

|库行|本轮A2 TP2结果|交付|
|---|---|---|
|3、7、8、9、10、16、19|两个owner复核通过；19的微小效果仅rank1可归因|五个独立补充ZIP；等待黄区审查和正式测试|
|12|实际采样消费链未证明，停止同原因重试|仅保留开发源码、CPU测试和失败记录，非通过补充件|
|32|执行、真实消费、恢复通过，但changed=0|保留无数值现象结论，不授予资格|
|34|社区视觉路径无合格BF16带bias候选，初始化后引擎报错退出|当前部署不适用，非安全拒绝、非两个owner通过|

本轮相关CPU回归54项通过；每个新补充ZIP在独立冻结源码解压目录运行其对应CPU测试，ZIP逐文件SHA复核通过。未修改冻结r7.4.1标签、包、src或fault_injection；未把蓝区结果登记为A3。

附件尚不能称为全部完成：上述三个参考项未达到数值现象交付条件；C3等待N2六条补提记录、C4等待D0/D1及噪声输入；C5正式登记和r8整合等待相应黄区测试。没有新输入时维持关闭，不伪造登记或发行。
