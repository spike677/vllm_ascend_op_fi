# Fault20 / Fault39 真实 Worker 验收与多模型进展

本轮基线为 `dcbef7a027bbc39e7796f12880305c3fa516c1f8`。只推进优先缺口，
没有重做17条普查。正向验收使用**完整预训练模型、安装后的wheel和标准创建的
vLLM Worker**。所有新原件留在124，公开文件只有实现、总结与校验信息。

## 结果

| 机制、实际架构 | 结果 | 实际范围 |
|---|---|---|
| Fault20 WeightNz / Qwen3-0.6B | B/F/OFF通过，统计首差通过 | 一次真实输出投影，BF16临时self尾数截断到3bit；原存储未改，完整logits变化且有限，首token不变 |
| Fault20 WeightNz / SmolLM2-135M，实际Llama | B/F/OFF通过 | 一次真实输出投影，完整logits变化且有限、OFF精确恢复；首token均为2、显示文本为空，不作为文本质量验收 |
| Fault20 QMM / 真正DeepSeek-V2-Lite W8A8 | 只读探索通过，故障资格待补 | 两轮各134次实际调用、134个实际模块，268次original Run；每个观测目标的signature/cache incarnation/layout/slots跨轮一致 |
| Fault39 MatMul / Qwen3-0.6B | B/F/OFF及统计首差通过 | 3种完整存储无损替换资格后，有限BF16双输入在真实MatMul产生151936个+Inf；token576→0→576 |
| GLM-4-9B-0414 | 预训练NPU正常推理通过 | 实际Glm4实现，2次各4token，8个完整词表logits有限且逐步重复一致；未注入 |
| DeepSeek-V2-Lite W8A8 | 预训练NPU正常推理通过 | 实际DeepseekV2、MLA+MoE，2次各4token，8个完整词表logits有限且逐步重复一致；未注入 |

Fault20和Fault39每个正向请求都只命中1次，覆盖范围是自动识别的一次自然输出
投影，**不是28层全部注入**，不是全部请求持续触发。完整模型被执行与完整模型
所有合格点被注入是两件事。新Worker仍需重新资格，不能直接复用上一次的地址、
request ID或owner票据。

QMM的实际head在量化说明中为FLOAT，不能将它标成QMM。本轮没有选择任意一个
中间候选冒充已验收的输出投影。下一步应给真实中间量化模块定义独立的输入、
输出、consumer和可信请求合同，再做lossless/B/F/OFF。268个观测事件不授予
全量分母或故障资格。详见[只读摘要](summaries/qmm-readonly.json)。

## 环境及构建

124物理卡5，910B4/A2，32GiB，driver25.5.0.b060；CANN8.5.1，
vLLM-Ascend0.18.0镜像，torch2.9.0+cpu、torch-npu2.9.0.post1+gitee7ba04、
Transformers4.57.6。实际镜像ID为
`sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`。
使用TP1、eager、单请求fresh prefill、同步调度，关闭prefix caching、chunked
prefill。没有图模式、TP2、decode故障、A3或其他vLLM版本的新增验收。

三个Python包在镜像内构建并安装到`/installed`；真实Worker记录实际导入文件
哈希。Fault20正式B/F和SmolLM2使用packages-a11/build-a8，native profile为
`6fa99506f19ec6df4fef4560a3e05560c5de5118c5cd35f24eec7800ffd320ff`。
Fault39正式统计对照使用packages-a7/build-a4，profile为
`df08dae9203b39821119acc238e8b2c4f244d91ec230108c1a17d0ce08fc8c0e`。
最终packages-a11/build-a8也单独完成Fault39的9阶段真实B/F/OFF回归，profile
`7854e4b804d679e7c06c92364384c73eeb0024675f13a0aed9d54785b7bc519c`；
见[最终构建回执](summaries/fault39-final-build.json)。这次没有替换早期正式统计
对照的原件或构建身份。提交源码与最终冻结包按打包的LF规则逐文件匹配，见
[源码匹配](summaries/deployed_source_match.json)。
构建和完整CPU556项回执见[summaries](summaries/builds.json)。Native CPU
测试包括current formals、request暂停/恢复、重复只读不重置fault预算、PTA路由
和真实ELF查找顺序。CPU结果不计入NPU通过次数。

## 自动选点和调用链

```text
Host真实enqueue ID → 检查EngineCore请求 → advance返回
→ 原Worker / runner execute_model → 完整模型forward
→ compute_logits的实际LogitsProcessor和head参数
→ quant_method.apply → 当前typed GWS / executor / 同线程Run
→ 原存储保护、真实输出、同步、资源释放
→ 全局词表logits → query → 下一fresh请求 / OFF
```

选点依据实际调用结构、类型、quant_method与唯一Native目标，不要求用户层名，
不按“第一个候选”或max_k选层。head作为实际参数观察，不虚构head.forward。
Qwen将head和Embedding绑定成同一个模块；SmolLM2有两个共享Parameter的模块，
已按真实别名关系分别处理。模型名不授予路径资格。

Extension不创建/停止/重启Worker，不替换Worker或runner的核心执行方法。
Host按`enqueue → advance → wait → query`逐请求推进；正常RPC返回后才由原
调度器运行。实际token/position字节、scheduler ID、Worker/模型/thread及
最后query hidden states均核对。阶段间Native暂停且无inflight/lease；失败
保留隔离，不能用注销owner冒充恢复。详见[接口和失败语义](../../docs/worker_extension/host_driven_replay.md)。

## Fault20具体干预与部署限制

干预发生在当前WeightNz真实Run前：读取完整self，构造同形状temporary buffer，
BF16尾数截断到3bit，真实kernel执行1次。完整原self和weight存储前后哈希相同，
临时输入读回与释放有回执。8阶段为discovery×2、reference、lossless、
qualification_off、fault、off_observed、disabled_off。末阶段无Native故障回调，
完整logits恢复；注销后健康请求也恢复。

Qwen默认绑权head为ND，并未自然进入WeightNz。通过范围明确要求：

- baseline前执行真实NZ格式转换，保持完整逻辑权重哈希和Parameter别名；不是故障时改权重值。
- 相关Embedding输出转ND，逐调用验证全部逻辑值完全相同；B/F/OFF共用。OFF参考的是这个显式NZ部署，不是默认ND部署。
- `VLLM_ASCEND_ENABLE_NZ=2`、`DEMONCAT_F20_REQUIRE_TYPED_GWS=1`。后者仅对两种WeightNz API关闭上层PTA快捷路径；真实OpExecCache仍执行。
- main/nn/math代理、实际dlsym胜出地址、状态符号、加载库和torch-npu wheel指纹均匹配。默认PTA路径仍待适配，没有把猜测的upstream commit当成匹配源码。
- overlayfs maps/stat身份不一致时，使用同镜像7个库的原字节私有副本只读挂载到容器原路径；没有放宽身份检查、修改主机Toolkit或提升容器权限。

CPU_VERIFIED manifest只是显式首次实机资格前提，没有自动提升成可复用
NATIVE_VERIFIED profile。实际scoped attempt与通用profile分别记账。

## Fault39具体干预

原生ABI v1保留，增加可选live-input guard、同generation discovery续接和
cache incarnation。当前self/mat2真实存储、M/K/N、offset、stride须与模型
参数完全相符。先做self/mat2/dual三种无损替换，每种都比较完整输入存储、真实
输出和全局logits。随后仅一次受控有限BF16双输入：self为正值，mat2每列采用
受控正负排列，真实MatMul产生+Inf。没有事后改写算子或logits返回值。

原self/mat2完整storage保持不变；真实输出151936个+Inf，首token从576（` The`）
变成0（`!`），OFF完整输出/词表hash和token恢复。这个受控数值范围机制不代表
自然prompt质量缺陷，也不自动等同于历史事故的原始触发条件。

## 原始 statistics/mix 的独立首差

`task=statistics, level=mix, precision=high, summary_mode=statistics`；完整
API/module输入输出，无算子过滤。采集包未被本实验修改，文件哈希入collector
identity；不是故障28的私有CPU-FP64采集修正。比较公式固定为
`abs(F-B) > 1e-8 + 1e-5*abs(B)`，保留原JSON精度。

| 机制 | 比较位置 | 首差、字段 | B / F | 保留不可观测字段数 |
|---|---|---|---|---:|
| Fault20 | 独立Worker同step5 | `Functional.linear.112.forward/output/0` `/Max`，索引3047 | 17.625 / 16.875 | 首差前6620 |
| Fault39 | 独立Worker同step6 | 同一API身份，索引3031 | 17.625 / +Inf | 首差前6128 |

Fault20绝对差0.75，相对约4.2553%。B自比分别3058/3042条，无可观测差异，
不可观测字段分别6642/6150。正式两组stack/construct、配置、collector身份
各自严格相同。比较器SHA为`28d9d87fa3bab5ed18c692b29e3ac880e86e46d31916397e0da040c4eb2e9bbe`，未修改。

比较器只读取三件套、配置和collector，不读Native命中或完整张量。上述是
**统计字段对比**，不是逐元素对比；首差在Python linear输出，而替换发生在
Python输入观察之下。四项统计不能唯一反推临时存储机制，需另结合运行时证明。
null仍不可观测，不能断言之前所有张量完全相同。原始统计不提供完整请求映射，
保留`UNPROVEN_REQUEST_MAPPING`，真实scheduler映射由另一份运行证据证明。

Fault20实采step2/5、Fault39实采step2/6；表示完整prefill资格阶段，不是decode
token，不重命名step0。此前同Worker不同phase的slot_indices首差失败原样保留；
正式对照使用独立Worker同phase，没有修改统计字节、调整容差或过滤该算子。

## GLM与DeepSeek

GLM公开权重来自[zai-org/GLM-4-9B-0414](https://huggingface.co/zai-org/GLM-4-9B-0414)，
固定revision `645b8482494e31b6b752272bf7f7f273ef0f3caf`；镜像站传输，所有
LFS SHA256/非LFS Git blob对照官方API校验。13文件约18.82GB留在124，没有执行
remote code。实际`Glm4ForCausalLM`，两次4token均` The sky appears blue`；
1613次实际Matmul Run被观察，Native始终关闭、无owner。

DeepSeek来自本机真实ModelSlim动态W8A8预训练权重，实际`DeepseekV2ForCausalLM`，
MLA+MoE；两次4token均`\nThe sky appears`，4490次Matmul Run。两个模型各8个
完整logits有限且重复一致；这是正常加载和推理验收，不是质量评测或故障资格。
不能将Distill-Qwen证据归入DeepSeek-V2/V3。

## 保留缺口与交付

17条逐项增量见[mechanism_delta.json](mechanism_delta.json)。15/17/18的严格
ALIGNMENT_MISMATCH、Fault26 rank1写前NaN及未知初始化来源保留。28、33等历史
实验不自动获得新Worker资格。所有机制的公共all_eligible仍未开放。
历史Fault39四个具体CANN版本的证据保留；本轮正向Worker仅实测8.5.1，不把结果
转授其他Provider、A3、8.0–8.4或其他模型。黄区实际内源`src/demoncat`仍不可得，
CLI/catalog/bindings正式接线待补。

源码入口：[serving_context](../../src/demoncat/runtime/vllm_ascend/worker_extension/serving_context.py)、
[serving](../../src/demoncat/runtime/vllm_ascend/worker_extension/serving.py)、
[serving20](../../src/demoncat/runtime/vllm_ascend/worker_extension/serving20.py)、
[Fault20资格](../../fault_injection/serving_qualification.py)、
[Fault39资格](../../fault39/src/demoncat_fault39/serving_qualification.py)、
[当前formals](../../fault_injection/native_adapters/qmm_current_call.inc)、
[原生request状态机](../../fault_injection/native_adapters/matmul_auto.h)、
[Fault39 Native](../../fault39/native/runtime.cpp)。

原件根目录 `/home/worker-positive-20261117`。Fault20正式B/F在
`closure-batch-a2/fault20-baseline`、`closure-batch-a2/fault20-fault`；独立报告
`fault20-pair-analysis-a2`。Fault39在`serving-pair-baseline-a1/fault39-serving-qwen-statistics`、
`serving-pair-fault-a1/fault39-serving-qwen-statistics`；报告`serving-pair-analysis-a1`。

```bash
python3 /home/worker-positive-20261117/compare_pair_124.py \
  --baseline-case /home/worker-positive-20261117/closure-batch-a2/fault20-baseline \
  --fault-case /home/worker-positive-20261117/closure-batch-a2/fault20-fault \
  --step 5 --out /home/worker-positive-20261117/fault20-independent-new-review
```

输出必须为新目录。GitHub只包含本次审查实现、总结与哈希，不上传新的原始
statistics/mix、完整私有张量、模型权重或vendor库。逐条原始数据重算需要访问
124；公开摘要不能替代全部原件。失败和回归见[失败账](FAILURE_LEDGER.md)。
