# r7.1 全量 Qwen 产品回归（2026-11-23）

冻结 r7.1 的 12 项 Qwen 适用能力全部通过，未改发行包、标签或 main。机制 8 只适用 MoE，本轮 Qwen 不适用；不能计为通过。蓝区组合：Ascend910B4/A2、vLLM 0.26 / CANN 9.1、社区 Qwen3.8 W8A8、TP2、eager。结果不代表 A3 或 DeepSeek，不自动获得免许可资格。

每项通过真实 HTTP 请求和公开 ProductClient，在同组 Worker 内连续完成两个独立 owner。原始回执由 review_product_mode 独立复核：真实消费者、eligible/attempted/applied/consumed/restored、failed=0、changed≤applied、区域恢复、资源清零、新请求金丝雀精确恢复。性能不作门禁。

|能力|结论|owner 数|rank|审查记录|
|---|---|---|---|---|
|15.alias_detach|通过|2|0、1|[review_15.json](review_15.json)|
|17.spatial_padding|通过|2|0、1|[review_17.json](review_17.json)|
|18.api_half|通过|2|0、1|[review_18.json](review_18.json)|
|20.buffer_substitute|通过|2|0、1|[review_20.json](review_20.json)|
|22.wrong_row|通过|2|0、1|[review_22.json](review_22.json)|
|26.fia_destination|通过|2|0、1|[review_26fia.json](review_26fia.json)|
|28.fia.denominator_replay|通过|2|0、1|[review_28.json](review_28.json)|
|33.paged_decode|通过|2|0、1|[review_33.json](review_33.json)|
|39.output_proxy|通过|2|0、1|[review_39.json](review_39.json)|
|39.matmul_range|通过|2|0、1|[review_39mr.json](review_39mr.json)|
|41.tail_column|通过|2|0、1|[review_41.json](review_41.json)|
|quant.batch_shared_scale|通过|2|0、1|[review_batch.json](review_batch.json)|

成功窗口累计 2217.257 秒（包含加载、启动、采集和清理；不是纯 kernel 用时）。各 window_*.json 记录绝对时间、控制器与隔离日志 SHA，全部确认隔离清理器停止。15/33 顺序共用最后一个锁窗口，但分别启动模型进程；同一机制的两个 owner 不重启 Worker。

Native 三项实际 capabilities 原件见 native_capabilities_raw.json：两 rank 均 PRODUCT_CANDIDATE / PRODUCT_ACCEPTANCE_ONLY / product_qualified=false；运行时 333 文件与冻结源码逐字节一致，详见 runtime_bytes_native_attempt3.json。attempt 中 a3 仅表示第三次尝试，硬件始终 A2。

15 的独立机理检查见 alias15_independent_review.json：两个 owner×两个 rank，共 768 次命中，ND 视图真实写入后原 base 保持旧值，真实消费者读取旧 base；初始非有限值为零。标注为“别名断开语义（ND 视图），不含私有格式触发条件”。

此前 33/native 两次被外部 stop/kill 结束，退出 137，无 OOM，不作故障通过或机制失败；成功重跑替代验收结论但保留失败记录。Native 的一次控制器同时消失，无法伪造其结束时间；当时为进程内有界线程，进程消失后无持久隔离设置。其他两个尝试仅资源锁冲突、未调用 NPU。卡2/5的外部租约依据授权、精确 PID/源码/锁路径复核后终止；未删除任何外部容器或数据，未改 Docker、驱动、CANN。独立派发器仍可重建租约。

负向范围：原回归实测缺许可、未知 profile、用户 Native 路径；未知机制99由公开配置/新 helper 的 CPU 测试覆盖，不能写成黄区 CLI 实测。蓝区没有黄区 python -m demoncat.cli 或其 daemon，此结果仅核对总单的库级调用及真实产品执行链。黄区鉴权、CLI 映射和负例仍需现场验证。

只提交审查摘要、原件 SHA 和窗口记录；完整私有 HTTP 回执、张量、隔离明细留蓝区，不上传 GitHub。
