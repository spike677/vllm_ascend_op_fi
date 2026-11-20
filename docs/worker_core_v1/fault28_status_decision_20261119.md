# 故障28：状态、研究边界与限时决策

2026-11-19夜只读审视；未新跑NPU。按后附文件的最新预算 **4小时NPU、两个工作日**，不是前一附件的8小时。优先RC2维护、0.26门禁、20/26/33普通任务，28只能用剩余窗口。不向黄区发送、不改冻结包。

## 1. 历史结果与原始身份

|变体/实验|路径|模型/几何/dtype/栈/硬件|原结论与本次解释|证据|
|---|---|---|---|---|
|28.fa.legal_tiling|HF SDPA接入FA，另有固定QKV算子回放|Qwen3-0.6B，B1/B8、S2048、Q16/KV8/D128、FP16，124 A2；早期包具体SDK以source_manifest为准，本次未重新审库|历史特定FA数值路径与传播；早期high_statistics包自身mechanism_accepted=false，不改为PASS|fault28-causal分支、native_fa_high_statistics包|
|28.fia.legal_tiling 子图|实验FIA→真实o_proj，TorchAir图|Qwen3-0.6B第2层QKV，B1/B8、S2048、FP16，CANN9.1.0，124 Ascend910B4|受控causal尾块机制通过；末query未变，不是完整模型/历史盘古复现|fault28-fia-graph分支|
|28.fia.legal_tiling 完整模型图|实验model_adapter，整图|Qwen3-0.6B，28层、B1/B8、S2048、FP16，CANN9.1.0，A2|30次图执行、840次FIA节点，首token logits传播；非vLLM Worker|fullmodel/device-grain/bf-only分支|
|28.fia.legal_tiling 完整统计|实验model_adapter，GE整图或eager|Qwen3-0.6B，B8、S2048、BNSD、Hq16/Hkv8/D128、FP16，CANN9.1.0，A2|图版内部记录为原生GE CSV及明确派生JSON；eager原生statistics/mix；合法M变化差异小，不满足后来的0.1%统计门槛|statistics-data与eager-statistics-data分支|
|28.fia.denominator_replay|实验model_adapter eager，不是vLLM|Qwen3-0.6B、B8/S2048/BNSD/Hq16/Hkv8/D128、FP16、M128/S2=128，CANN9.1.0、torch2.10.0/torch_npu2.10.0.post4/transformers5.5.4、A2|受控首块分母重复累计通过；三个step首FIA output Norm 660.572120626524→655.3088952891358、0.7968%；不是历史合法切分复刻|f879af9b，implementation/model_adapter.py及reports/first_difference.json|
|真实TND denominator候选|vLLM Worker真实调度|Qwen3-0.6B、TP1串行eager、TND2048、FP16、CANN8.5.1、vLLM-Ascend0.18.0、124 A2|**部分通过：只有正常关联，F/OFF未通过**|Source123，下面身份表|

远端分支已用ls-remote核对，本地git对象只读，不重写旧证据：
- [fault28-causal-20261029](https://github.com/spike677/vllm_ascend_op_fi/tree/96aa4d874fa296a7f2a489ac931f75250950fd15)：`96aa4d874fa296a7f2a489ac931f75250950fd15`。
- [fault28-fia-graph-20261106](https://github.com/spike677/vllm_ascend_op_fi/tree/bf891257cdd0f8e09e5ad50a39160fdb94d12472)：`bf891257cdd0f8e09e5ad50a39160fdb94d12472`。
- [fault28-fia-fullmodel-20261109](https://github.com/spike677/vllm_ascend_op_fi/tree/2aa4432ce92a2c878f6fb6ce3a648b1049df4b48)：`2aa4432ce92a2c878f6fb6ce3a648b1049df4b48`。
- [fault28-fia-device-grain-20261109](https://github.com/spike677/vllm_ascend_op_fi/tree/19ba4636ae822b0274b5728b1e652d01e1e9756d)：`19ba4636ae822b0274b5728b1e652d01e1e9756d`。
- [fault28-fia-bf-only-20261110](https://github.com/spike677/vllm_ascend_op_fi/tree/6dad85c480255829b0a4e74657a7e2ff2ce390b3)：`6dad85c480255829b0a4e74657a7e2ff2ce390b3`。
- [fault28-fia-statistics-data-20261110](https://github.com/spike677/vllm_ascend_op_fi/tree/d369d06a13b5e04e92aba8d73e7bc14c15dff98c)：`d369d06a13b5e04e92aba8d73e7bc14c15dff98c`。
- [fault28-fia-eager-statistics-data-20261110](https://github.com/spike677/vllm_ascend_op_fi/tree/13217d6c6b97e77e6857d92615ce5b28c433160b)：`13217d6c6b97e77e6857d92615ce5b28c433160b`。
- [fault28-denominator-replay-data-20261113](https://github.com/spike677/vllm_ascend_op_fi/tree/f879af9bd19a4c1ca099c81c515e4c33cbee20c7)：`f879af9bd19a4c1ca099c81c515e4c33cbee20c7`。

`artifacts/attention_tiling` 的历史验收包位于上述历史提交中（当前精简发行树不含全部大件）：

|原位置（相对仓库）|提交|实际文件SHA256|
|---|---|---|
|artifacts/attention_tiling/fault28_causal_matrix_statistics_20261029/manifest.json|96aa4d87|db24066e8265bf44b457030e935b2c9dc8a8787dee92312601a0bb52159df7c4|
|artifacts/attention_tiling/fault28_native_fa_high_statistics_bf_20261029/package_manifest.json|96aa4d87|561221711e723c891f2feb2729720f9a761e5452e42402ebf1c27f697f229eb6|
|artifacts/attention_tiling/fault28_native_fa_high_statistics_bf_20261029/source_manifest.json|96aa4d87|012687d1d5e9c2d908e8bc6adad65b08a4e38e7c1e59d8d18a3db520000f24b0|

SHA由git show原始blob重新计算；package_manifest自身mechanism_accepted=false被保留。FIA图包在bf891257的artifacts/attention_tiling/fault28_fia_graph_statistics_20261106，完整模型包在2aa4432c相应artifacts目录；这些不同包不能用上表FA包的SHA代替。数据分支README、SHA256SUMS、ORIGINAL_BYTES与ZIP各自保持原字节，不重新打包历史数据。

## 2. 为什么普通入口关闭

legal两个变体是HISTORICAL_ONLY，本轮不扩展。实验denominator取得了固定BNSD输入的机制和prefill传播证据，但模型适配器与真实vLLM scheduler、live KV/Execute生命周期不是同一路径。实际调用关联不等于故障替换合法，不等于默认/private数值相同。因此保持TND_FAULT_IMPLEMENTATION_INCOMPLETE；不从旧PASS、命中日志或统计首差签发Worker资格。

历史三变体作为研究保留，不属于交付；研究人员需显式create_research_provider或研究类，公共FaultConfig/registry/ordinary入口不授通行。research_entry与request_plan为RC2.2待实现接口；本页不声称现有字段已上线。

## 3. TND已完成与未完成

Source123只证明当前public tensor对象、typed描述符、Slice/FIA实际launch、ELF来源及11条RPC/14条Native事件对应，解除后同Worker健康请求完成。完整stock/private数值等价、M128对照、F/OFF仍未证明。

|身份|值|
|---|---|
|Source123清单SHA|a72d5feffef860028f3d0e10a72799c80c20fafc90f827c66c4f6b4413107524|
|Source122候选build completion SHA|58dfd1021a8a77267ab94e2a3b17d0a913c22365d4f0e7d2d84ba49473485181|
|实际FIA key|5000000000000200103|
|实际FIA注册ELF SHA|d99ce5e9435a3df59ff4432f53f997305638de40e4b306918f587faf8d96dd81|
|Source127独立完整报告SHA|e0684e5479b74ddadf47ea64d32a25ea1211429510f5aa718bbd46f40ec3d942|

见[tnd28_source123_normal_review.md](tnd28_source123_normal_review.md)。生产libnnopbase完整源码与依赖源码相等未证明；Slice POD未解码；LSE临时占位生命周期限制保留。新B1必须新建自己的收据，不能把上述旧build重标为Execute借用构建。

Execute借用：开发线已有974f3402整合的opt-in/取消合同，仍未授F。另一个tnd28-denominator-product工作树存在7项修改、2项未跟踪（包括attention28_typed_plan.py）；尚未审入主线，不能当作已发布实现。下一步先看真实include/aclnn SDK构建与取消/释放合同，不用opbase头替换所需接口。

## 4. 迁移价值与边界

可复用的是独立所有权、真实调度绑定、资源回读、对照/恢复和证据格式。不可直接移植的部分：固定8.5.1 SDK安装路径与头指纹、私有nnopbase executor/rt2Tensor描述符布局、Contiguous child workspace offsets、GetWorkspaceOffsets/RunWithWorkspace及abort释放语义、特定生产libnnopbase ELF及两条安装链接、arch22 tiling/key/serializer/kernel模板。

CANN9.x需要重新识别实际ABI、构建、stock/private桥接和F/OFF；版本号大于8不能自动放行。vLLM0.26还涉及模型方法、scheduler输入、attention元数据与缓存接口变化，先受0.26空间/驱动门禁约束；0.32未批准拉取/实测。A3要实际A3编译/执行及新登记，A2不授资格。

负责人描述的Qwen3.8混合Attention/FULL_DECODE_ONLY与这里TND2048 prefill不相同，需要现役inspect确认各分支，不能仅改模型名。DeepSeek V4.1 MLA的Q/K/V维度、投影/压缩KV、kernel及缓存语义亦不相同。当前二者均未适配验收；本轮不跑DeepSeek，GLM不在现役范围。

## 5. 限时计划和进度

限额4小时NPU；自然期限两个工作日，从本轮B1实际开始计时并记录绝对起止时间，禁止自动延长。当前仅只读审视，B1尚未开始、NPU用时0、剩余14400秒。每步记录controller起止、占卡秒数、失败原因及作用域，累计包括失败和控制请求，不只计算F时长。只有空闲窗口，RC2/其他普通任务优先。

|步骤|前置条件与执行|预期阻塞/止损|当前|
|---|---|---|---|
|B1|审查已有借用改动；固定8.5.1镜像真实include/aclnn构建；新收据/依赖SHA；无设备合同|SDK声明/取消接口不匹配，同因两次停止|未开始，0秒|
|B2|B1通过；真实Worker正常请求借用与取消，F关闭；完整输出与stock/private或M128对照相等；全资源释放|当前对象/descriptor/workspace生命期、数值桥接；同因两次停止|未开始，0秒|
|B3|B2通过；自动单目标真实FIA launch一次分母replay；读Attention/hidden/logits|真实执行准入或分母模板语义；失败不重跑F覆盖|未开始，0秒|
|B4|同Worker OFF完整hidden/logits精确恢复、无故障hook的health、全lease/hook/borrow/inflight0|任何恢复不确定都保留owner与失败证据|未开始，0秒|
|B5|同Worker新Runtime owner/provider第二作用域；request_plan与实际阶段对应|两个独立作用域不等于重启Worker；相同失败两次停止|未开始，0秒|

预算用尽或止损后交负责人决定；不扩大预算。只有独立审查通过才登记精确A2 BLUE_COMBINATION并进入RC2.2；黄区默认关闭。不声明其他几何/新CANN/A3/现役模型。

## RC2.2 边界补充

负责人已接受后续B2的stock/private完整哈希桥接，限制为stock census FUNCTION_BIND_NOT_UNIQUE及health仅token级。B3–B5在开发线继续；本RC2.2仅保留原研究入口，公共入口关闭，不包含新的Native两阶段绑定。M128控制已止损，不再重跑。
