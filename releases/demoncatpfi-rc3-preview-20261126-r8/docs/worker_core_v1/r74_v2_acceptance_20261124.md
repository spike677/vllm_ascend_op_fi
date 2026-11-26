# r7.4 v2 示例验收（2026-11-24）

组合仅为 BLUE_COMBINATION：A2 / Ascend910B4 / vLLM 0.26.0 / vllm-ascend 0.26.0rc1 / CANN 9.1.0 / Qwen3.8 社区 W8A8 / TP2。不是 A3 验收或免许可登记。

|示例|同组 Worker 两个 owner 的每 rank 结果|恢复与负例|
|---|---|---|
|weight_scale_half|自动选中真实 dense QMM 的 BF16 weight_scale；每 owner 写入一次、48 元素改变、4 次 QMM 读取|权重原字节及原派生对象恢复、资源清零、hidden 金丝雀精确恢复；缺许可 exit4。没有同输入 fault hidden 传播证据，不作传播结论|
|kv_value_cell|16 层实际 full attention，每层一个真实 V cell；每 owner/rank 写入16次、16元素改变、48次后续 FIA 读取；同输入 fault hidden 改变|实际页回收前字节恢复，租约/资源清零，hidden 金丝雀精确恢复；缺许可和超范围 exit4；额外真实 HTTP 流断开已观测 FINISHED_ABORTED，恢复通过|

KV 正例八项计数每 owner/rank：observed=eligible=attempted=applied=changed=consumed=restored=16，failed=0。changed_elements=16；48次读取单独记 reader_observations，不伪装成48次写入。已修改的请求后续 decode 读取污染值是预期现象，不要求它恢复到未注入结果；新请求的金丝雀须精确恢复。

最终 KV 原回执 SHA256：`ca73ddbb02ff093d4d4611b37ec204b08c7a608d9a2ead2547c83e3df5718d33`。独立审查见 `r74_kv_final_review_20261124.json`；完整回执在蓝区私有工作目录，不进 Git。取消负例的两 rank 通知序号、请求代际、实际状态和恢复证明交叉核对，不能只相信 actual_abort_proven 标志。

## 接线与修正

首轮在写入前被 `KV_WRITER_SLOT_NOT_OWNED` 拒绝。实际混合注意力的 allocation block=1536 tokens，kernel block 按真实 cache 和 BlockTable 读取；物理页面被拆成多个 kernel block。已按实际 `physical_block_size / block_size` 映射并核对运行时比例、group及DCP范围；不是放宽页面归属检查。CPU覆盖外部页面、错误比例和后续同区域合法写入拒绝。

备份和恢复复用 RegionLeases；强引用保留分配身份。每次真实 cache writer 之后核对被持有区域字节，允许其他slot合法写入但不覆盖它们。消费者证据来自真实 FIA 的相同存储、block_table、实际长度和读取完成事件。EngineCore 在每次执行前通知页归属，在实际 free/pop_blocks_for_free 之前做全 rank 恢复确认，避免等下一轮 finished_req_ids 时页面已复用。

HTTP取消通过实际调度状态 `FINISHED_ABORTED` 证明。尚未读到故障值就取消时，只有同请求代际、同stream完成、真实释放前边界及期望字节一致才允许回滚，consumed保持0，标记 `ABORT_BEFORE_CONSUMER_BYTE_ROLLBACK`；不把它当成功注入验收。当前实机取消发生在两个前向之后；未消费取消回滚分支由CPU覆盖，不能宣称所有取消时序已实测。

## 支持边界

仅专用 network-none 实例、唯一受控 HTTP 提交者，B1、4生成token、16层 full attention、prefix cache关闭、无connector/投机/异步调度、PP1。GDN状态不属于此KV示例；MLA、共享页、多请求、异步/其他stream和未知源码拒绝。后续部署必须提供真实入口排空和页释放绑定，不能照抄返回True的回调。只读页面普查不是写入验收。

原有 r7.3.1 Runtime/机制共享代码保持逐字节一致；wheel版本升级只改打包元数据，逐文件差异单独披露。v2仅可信注册的 persistent_* site进入新路径，普通公共配置不能绕过门禁。33只附采集器v2，A3注入资格仍关闭。

## 窗口与隔离

见 `r74_v2_windows_20261124.json`：页面只读窗口254秒，首次KV写入尝试256秒（归属映射拒绝），修正后正例264秒，最终正例与取消负例274秒。各窗口均有绝对UTC时间和精确设备docker-stop记录，结束后有界清理器停止，隔离解除。没有修改Docker入口/派发配置，没有删除或重启其他任务容器。服务退出不替代同进程恢复证据。
