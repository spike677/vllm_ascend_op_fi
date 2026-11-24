# 2026-11-22隔夜推进总结

## 发行与维护

r5已冻结发布，源提交 `d4ae552df2b4fd82641bb39f234d2071dcccaed5`，发行提交 `a76f33cf`，标签 `demoncatpfi-rc3-preview-20261122-r5`。三wheel版本、源码包、SHA清单、无设备安装回执见 [交付单](../../releases/RC3-PREVIEW-R5-HANDOFF-20261122.md)。1071项CPU回归0失败0错误、2旧研究skip；随后56项有重叠的最终检查通过，不累计为独立测试总数。124隔离无设备安装378源文件逐字节核对通过。r4/更早冻结标签和包不改动。

r5包括900秒正式阶段上限、600秒预热/最多6次、39mr的512KiB内证据修复、逐项套件与失败恢复、模型制品和调度身份、五项产品mode=all。产品CLI与HTTP金丝雀的 [RPC契约](product_rpc_contract.md)齐备。身份不足的r4文本只保留结果，不能登记；等待r5完整回传。目前未发现新的黄区r5材料，不向黄区直接发送消息。

## 33方向A：公共入口通过限定验收

组合：124 Ascend910B4/A2、vLLM0.26.0/vllm-ascend0.26.0rc1、torch2.10.0+cpu/torch_npu2.10.0.post4、CANN9.1.0、社区Qwen3.8 W8A8、TP2、单请求2304-token prefill+4生成token。不是A3或其他形状验收。

实际decode launch为key5100000000010200203、10核、每KV头5组S2=512。控制实现保留BF16矩阵操作数、FP32输出与累加、BF16概率和最终输出；仅分组改512→256。旧FP32操作数/CPU控制的全模型失败保留，没有下调99%门槛、没有按元素挑选写入。

|正式验收|rank0|rank1|
|---|---:|---:|
|每作用域observed / eligible|64 / 48|64 / 48|
|应用/消费/恢复|48 / 48 / 48|48 / 48 / 48|
|失败|0|0|
|最低control逐位一致率|99.9348958%|99.9674479%|
|分组自身变化元素（control→alternate）|7262|6764|
|control对native的残余元素总数|8|2|
|hidden/logits变化、OFF/health精确、资源零|通过|通过|

同组Worker两个作用域均通过；16次prefill/作用域/rank明确不适用，全部48次decode符合条件且持续触发。真实消费者为AscendRowParallelLinear，注入区字节与租约独立恢复。两个作用域耗时1.95192×、1.74387×，不作性能门禁、不追加优化。1152个decode、384个prefill实际launch覆盖两进程，未触及采集上限，所解析模板字段匹配。

[独立报告](paged33_direction_a_20261122.md)说明99%仍不是100%原生指令等价；实际模型传播不能全部归因于分组，少量控制残差单独披露。仅在开发线登记两个实际rank的BLUE_COMBINATION身份，仍需可信许可；未纳入r5，不授予产品HTTP/A3/TP1/MLA资格。A3需要独立launch/固定QKV检查点/两作用域恢复。329运行源文件审查：325字节相同，4个仅CRLF、AST一致，实际运行SHA完整保留。

## V2-Lite参照与DeepSeek准备

V2-Lite W8A8/A2/0.26/CANN9.1/TP2、EP关闭：AllGather输出行错位参照同组Worker两作用域通过。每rank104 observed、26 eligible、26应用/变化/消费/恢复、failed0；78次单行decode不适用。真实aten::add读取，OFF/health精确、资源零，耗时1.13196×/1.07648×。这是 [AllGather输出代理参照](allgather8_reference_20261122.md)，不是MC2/EP或V4资格，也不覆盖历史反排列变体。

V4-Flash-0731 W8A8/A3/0.32rc1/CANN9.2.1/TP8+EP/DP1/8卡/≤4096单chunk：已完成 [固定1.5噪声规则与判定器](dsv4_noise_acceptance.md)、只读标定采集、[只观测入口与源码清单](dsv4_observe_handoff_20261122.md)。CPU测试通过，不是V4实机通过。Qwen精确恢复标准不变，字节恢复/资源零不接受噪声放宽。V4实际源码SHA、同进程及新进程K≥5标定、只观测计数尚待黄区；缺失项不填默认值。W4A8 NaN仅诊断，V4.1/1M不再设计实验。没有蓝区V4 NPU运行或第二批资格登记。

## 抢占与隔离记录

33新限时13:00–19:00 UTC、最多7200秒NPU窗口。10个窗口实际合计2621.496682秒（43.69分钟，含失败/加载/清理）；另有预算前census529.438秒单列。正式公共窗口14:12:39.648610–14:20:23.565179 UTC，463.916598秒。AllGather另用307.185989秒。所有窗口已结束，有界清理器停止，派发能力恢复；没有更改Docker入口/daemon/驱动/CANN，没有停止外部容器或删除外部数据。详见 [全部33窗口](paged33_direction_a_20261122/windows.json) 与 [AllGather窗口](allgather8_reference_20261122/window.json)。

进程退出有vLLM强制收尾Worker与Python shared_memory警告，原日志保留；不把故障资源清零扩大成第三方进程无告警。私人完整QKV、host指针参数和大回执未上传Git。

## 开发提交与待交接

- `3b53801d`：V4噪声规则、只观测与标定入口。
- `fc77388f`：AllGather参照与独立复核。
- `6dbcf6c6`：33真实边界控制、typed binding、实验检查点。
- 本报告同提交：33正式公共复核、严格逐调用记录计数、BLUE身份登记与源码/窗口总结。登记、错误身份拒绝、无许可拒绝及审查器相关17项CPU测试通过；与前序测试有重叠。

当前能交付的蓝区实施与检查点已完成。下一批V4适配、A3资格和product_closed依赖黄区实际回传；由负责人转交现成文档，Codex不代发、不补造结果。最新12项状态见 [进度总表](continuous_progress_20261121.md)。
