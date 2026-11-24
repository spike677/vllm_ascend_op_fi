# 黄区文本接收：RC2.1、DeepSeek与Qwen摸底

来源：负责人转交的openchamber文本，接收目录 `D:/CodexArchives/20261121/yellow-rc21-deepseek-qwen-results`。原件保留Git外；本记录不复制机器地址、卡号、容器或内部模型路径。

原文20,184字节，SHA256 `968bdb5ecd16886c5f19c979718c1f75f8f76bf09b11b9e7b9cdb1e9d0dd0e62`，已与归档manifest独立核对。证据等级为黄区文本回传，不能当作蓝区原件审查。

## RC2.1回归与缺陷分类

- 回传master升级提交03c7245；18/39/22在Qwen3-0.6B BF16、A3、0.18、TP1测试部署各完成两作用域。报告安装检查issues为空，原件留黄区。本轮不重做TP1，不改冻结RC2.1。
- 登记运行读取器CANN8.5.1，toolkit版本文件8.5.0，保留两来源，不强行合并。回传包含源码/清单SHA，但不是安装检查完整JSON；本记录仅登记回归回传，不另作新的组合准入。
- 18 summary缺元素级效果；22仍只有request ID；39 `effects.recovered=null`而`recovery_confirmed=true`。定性为摘要字段/契约缺口，阶段1统一修正，不能把null自动当作false或编造未采到的元素数。
- 默认已登记测试组合的permit=null与临时验收许可缺失不是同一概念。未登记RC3组合仍需部署许可。

## Qwen3.8摸底

A3、0.26系列、TP2纯文本实例可运行；本批尚无新持续故障公共入口两作用域验收或TP2通信跟踪。历史TP1结果只保留，不安排新TP1实机。单/多卡token一致只代表该提示的输出，不替代完整hidden/logits或故障恢复验收。

## DeepSeek V4.1 W4A8

黄区TP8+EP、DP1、eager、无异步/prefix/speculation、max_seqs1、32K、Engram CPU offload实例可运行。32K关闭chunked prefill时KV预算不足；统一采用开启chunked prefill并确保实际请求只占一个chunk。回传max_num_batched_tokens=4096仅证明5-token短提示可以单chunk，不能声称任意32K提示均单chunk；仍需核对实际调度及请求长度。

每rank报告：SparseFlashMla40、LightningIndexer8、独立DynamicQuant92、RmsNorm129/RmsNormCast40、InplacePartialRotaryMul136、MoeDistributeDispatchV2/CombineV2各40、QuantMatmulWeightNz128、MatMulV2 138。计数是prefill及run2-run1推导的第一次decode，不是直接证明多token持续覆盖。

- 8没有MoeTokenUnpermute，适配dispatch/combine动态映射。
- Batch只覆盖独立DynamicQuant92，不纳入融合GroupedMatmul量化。
- MTP未执行；纯文本视觉算子为0。缺图像预处理配置，不能为17新增准入。
- inspect模块列表7个rank返回、1个未返回；其他只读inspect被串行TP1门禁拒绝。应分离只读门禁并报告缺rank，不能放宽执行准入。
- 跨进程固定请求输出不一致，后续同进程结果见另一份接收记录。没有完整软件栈/全部源码指纹/实际dispatch映射/完整数值数组，因此这里只能支持现役摸底和问题定性。

待补材料由负责人转交，不直接要求黄区操作：确定性设置及同进程完整输出检查结果；缺rank原因；实际稀疏key/EP dispatch字段；必要的完整摘要与安装检查文本。当前`product_closed=false`。
