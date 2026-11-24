# Batch共享量化torch变体：全rank持续实机验收

组合限定：124 A2 Ascend910B4、vLLM0.26.0+empty、vllm-ascend0.26.0rc1、CANN9.1.0、TP2、eager、B8、4生成token。当前为可信实验site接线，不授予公共Runtime/RC3组合准入。原Native变体及冻结发行物未改。

## 实现和可复核证据

自动发现模型各`AscendW8A8DynamicLinearMethod`，绑定真实`apply`调用，源码SHA固定为`ac9bd6ec05356b639665ec05ae21b19e1d0f0dc6723f7568b2f0ab70821285de`。

原DynamicQuant执行后，备份原INT8 Q与FP32 scale；当前调用全部真实行采用一个`absmax/127`。新Q/scale写入原输出缓冲区，保持dtype/shape/layout，交给原生`npu_quant_matmul`。真实QMM入参必须是这两个对象；同stream事件确认原QMM完成后恢复，写前/恢复后SHA一致。没有修改权重、输入或QMM返回值。未覆盖的独立DynamicQuant调用、padding/请求映射不明、参考值不一致都拒绝验收。

两作用域各为normal/fault/off/health；每批8个实际内部request ID，prefill一次、decode三次。预热上限6批，连续两批全有限且完整hidden/logits一致才进入验收。各rank独立触发，命中处没有全组通信。

## 跨设备算术边界与修正

a1两模型首次故障都被`CPU_REFERENCE_QUANT_MISMATCH`挡住，未记通过。a2简单数值诊断没有复现；a3固定种子24组BF16矩阵中一组出现70个Q元素不同，而scale逐位相同。此证据只定位到设备算术路径与CPU参考的Q差异，未追查第三方kernel根因。

a4明确使用**torch CPU-FP32参考重算**，再将Q/scale传回原NPU缓冲区；每次仍以独立NumPy参考逐元素核验。这不是“纯NPU重算”，也未降低阈值或更改共享尺度公式。每个事件记录`arithmetic_backend=torch_cpu_fp32`。输入临时拷贝仅在内存中计算，不写文件或上传完整张量。传输和观测开销计入性能结果。

## Qwen3-0.6B W8A8：a4已通过

84个实际匹配投影，每rank每作用域336次调用；两个作用域、两rank总计1344次。observed/eligible/attempted/applied/changed/consumed/restored全部336，failed=0。实际原QMM消费、CPU参考、逐调用字节恢复、owned资源清零、OFF/health完整四次hidden/logits精确恢复均通过。同组Worker身份保持不变，两个rank的hidden/logits均传播差异，生成token也变化但不作为门槛。

两作用域故障/正常耗时2.8584×、2.7968×；故障批次1.3349s、1.3240s。这是带完整观测和CPU参考的短提示性能，不能外推长上下文。

回执SHA：`13c9e00e99363154e3423bab09122db148f786df69f8ebe32cf4553169f37d35`。

## Qwen3.8 W8A8

a1无注入观测发现304个实际匹配投影，不是先验208。首批输出有非有限值，第2/3批全有限且完整输出一致；随后故障被上述算术校验拒绝。本轮保留首批异常，未修改GDN或宿主软件栈。a4在独立新进程中重现首批非有限、第2/3批稳定，然后完整验收通过。没有用小模型结论替代。

每rank每作用域1216次（304投影×4个forward），两个作用域合计4864次；observed/eligible/attempted/applied/consumed/restored全部1216，failed=0。rank0 changed=1214，rank1 changed=1213，两个作用域相同；其余调用是实际应用但量化结果未新增变化，不能删去或改成未命中。每次独立CPU参考与原QMM入参验证通过；每次Q/scale恢复SHA一致；owned hook/lease/inflight清零，两个rank的OFF/health完整hidden/logits与normal精确相同。

匹配投影构成为in_proj_qkvz/in_proj_ba/out_proj各48，gate_up_proj/down_proj各64，qkv_proj/o_proj各16。覆盖所有实际匹配的独立W8A8线性量化调用，不能声称覆盖融合进其他算子的量化。

故障/正常耗时4.3012×、4.2694×；故障批次6.2235s、6.1221s。两rank hidden/logits变化，生成token也变化。建议当前短提示B8、4token的实验请求上限30s；不据此批准任意长度或批大小。尚未授予公共Runtime准入。

Qwen回执SHA：`f47a064b6b22f356bbb29ad34f191511e9bde0677ff6c920a8b4d0d2e0f19395`。结构化总结见 `experiments/worker_core_v11_20261118/continuous_batch_real_model_summary_20261121.json`。

## 文件与资源

- `fault_injection/engines/dynamic_quant.py`：共享尺度公式与显式算术设备选择。
- `fault_injection/sites/batch_quant.py`：真实量化/QMM边界、消费、按对象备份与恢复。
- `experiments/worker_core_v11_20261118/continuous_batch_probe.py`：真实TP2、B8、两个作用域与预热。
- `tools/review_continuous_batch_site.py`：独立复核覆盖、原QMM、内部请求映射、两作用域同Worker、字节及OFF/health。

a4 source manifest为`e2bf45e944b240360ed3dd21f248a19c2416d964392f9f6cea2b73a0e8a7244c`。所有attempt用独立目录，原始回执不提交Git。卡2/5使用原资源锁及有界可逆清理器，不改Docker入口/驱动/CANN，不影响其他卡。

a1失败后Qwen容器卡在自身退出，已在FAILED回执写出后停止本项目容器；退出137来自此收尾，不记成外部冲突或OOM。原case1退出2；窗口483.475s并已恢复隔离。a2诊断17.995s、a3诊断16.026s，均已恢复隔离。a4窗口436.449s；四窗口合计953.946s，包含启动、加载、预热、诊断与退出，并非纯kernel时间。

隔离记录（北京时间2026-11-21）：a1 18:11:05–18:19:09，a2 18:19:39–18:19:57，a3 18:20:42–18:20:58，a4 18:21:49–18:29:06。全部记录cleaner停止并join、派发能力恢复；实际没有停止外部容器。原始退出日志仍有vLLM宽限期SIGTERM与shared_memory清理警告，未掩盖。owned注入资源清零在活着的Worker内完成验证。

本轮新增CPU检查10项通过：Batch site/恢复异常路径、共享公式与独立舍入参考、Batch回执拒绝项，以及M4动态EP归属校验；CPU结果不替代上述NPU验收。
