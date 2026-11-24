# 2026-11-22 本轮进展与剩余项

## 开发提交同步

开发线 `codex/product-integration-20261119`：17视觉验收提交 `7cb1ee08`，Native41提交 `0cd2da56`，26/20公共验收及MC2 CPU准备提交 `48a2ac29`。提交内容为实现、测试与摘要；原始完整回执和模型张量保留在私有暂存。12项×环境状态见[总表](continuous_progress_20261121.md)。

## RC维护与发行包

RC3预览r3已冻结六项：28、Batch、39 output_proxy、18、22、39 matmul_range。源提交 `112376eeb97f8fd6523851496424814aab45209a`，发行提交 `0a10cb2e7aef838402446bb50dd107718183e82e`，标签 `demoncatpfi-rc3-preview-20261122-r3`。

正式包身份以[交付单](../../releases/RC3-PREVIEW-R3-HANDOFF-20261122.md)和冻结标签为唯一依据。17、41、26、20只在后继开发线，不在冻结r3里。未发送黄区消息，也未修改任何冻结标签或包。r3可由负责人转交；新增候选仍需下一发行的源码、安装和包校验。

## 逐项蓝区闭环

以下均为A2 Ascend910B4、CANN9.1、vLLM0.26/vllm-ascend0.26rc1、Qwen3.8社区W8A8、TP2公共入口，同组Worker两个作用域：

|能力|每rank每作用域eligible|消费/字节恢复与OFF/health|fault/normal|
|---|---:|---|---|
|41|48|通过；16次直接KV prefill调用不适用|1.01725× / 1.04375×|
|26 fia_destination|64|通过；实际FIA数据写入当前residual|1.11659× / 1.12412×|
|20|1216|通过；真实DynamicQuant读取覆盖输入，原QMM消费Q/scale|2.36869× / 2.39569×|

所有eligible的failed均0，四token、两作用域、资源清零及负向检查通过。20首轮证据容量失败保留，修复为全部逐调用检查后的站点汇总，没有放宽容量或恢复条件。33仍因最终模型传播未达标关闭，不人为放大误差；17视觉已按真实encoder调用通过。A3/TP1不继承这些蓝区结论。

四个Native窗口合计1366.521秒，含模型启动、加载和清理，非纯kernel时间；均已解除卡2/5有界隔离，实际停止外部容器0个。见[窗口记录](../../experiments/worker_core_v11_20261118/native_windows_20261122.json)。第三方vLLM关闭警告保留。

## DeepSeek残留与兼容方案

V4.1仅做0.32源码与CPU准备，没有蓝区V4.1 NPU运行。稀疏MLA参数解析、真实config夹具、MC2 dispatch/combine对象关联已有测试；实际runner分片请求映射、接收端expert行归属、融合18/22/41接线未完成，不登记为可执行。

[A/B/C设计](dsv41_residual_acceptance_design.md)等待负责人结合黄区无注入对照选定。尚缺真实状态对象及初始化接口的运行身份，不能用递归清零代替安全隔离；不修改第三方kernel，不发布最终DeepSeek验收步骤。V2-Lite通过也不代表V4.1通过。

## 历史测试欠账与性能

[扩展RC3子集](native_cpu_review_20261122.md)1023项、0失败/错误、2项旧研究入口skip；随后审查器清理和登记更新相关26项通过。这不等于全部历史欠账消失。性能记录为实测值，未在负责人设定目标前擅自设新门禁。Native吞吐优化仍应保持每次消费完成、精确字节恢复和对象身份检查。
