# FA 全模型逐层、逐请求覆盖验收

[实现方式、真实调用链与关键函数](analysis/IMPLEMENTATION.md)

本文为 M/nRatio 联合切换实验。仅改变 M、实际 nRatio 始终为 8 的模型传播已在
[2026-11-09 的独立 B1/B8 补验](../fault28_m_only_model_20261109/README.md)通过，两轮 M-only 稳定、OFF 精确恢复；
对应证据为新采集，不以本文 joint 结果替代 M-only 归因。

状态：2026-10-31，124物理卡5完成完整 HF Qwen3-0.6B 的 B1/B8 验收，
联合结论为 `PASS_MODEL_ALL_ELIGIBLE_NATIVE_STRATEGY_COVERAGE`。
**本次矩阵内所有符合条件的受控 prefill 调用均实际切换：112/112，覆盖全部28层和8个请求。**
卡2外部推理服务保持运行，自有容器已清理，设备锁已释放。

| case | 请求数 | 全部SDPA | 受控prefill实际切换/符合条件 | 受控decode透传 | 受控layer/request映射对 |
| --- | ---: | ---: | ---: | ---: | ---: |
| main_b1 | 1 | 280 | 56/56 | 56 | 56 |
| main_b8 | 8 | 280 | 56/56 | 56 | 448 |
| 合计 | B1使用B8中的request_00 | 560 | 112/112 | 112 | 504 |

全部五轮共有280次符合证书形状的prefill，其中112次受控切换、168次calibration/native/off保持原生策略。
另有280次Q1/KV2049 decode，全部明确列为范围外并原API透传。B8一次调用承载8个请求，
504个映射对不是504次kernel调用。没有遗漏、重复、UNKNOWN或游离attention dispatch。
两组整日志各独立枚举280次attention执行，280个prefill的实际raw/POD再从原始日志逐条重解析。

正式attempt为`formal_stream_a1`；联合报告SHA256：
`16e82c3e072442839465a4e2b604d891430ead77bc2aa018a8f7ce9fe6fed8cf`。
原始统计、运行/调用/存储元数据及报告见[公开证据子集](../../artifacts/native_execution_strategy/fa_model_coverage_statistics_20261031/README.md)。
公开包共2,438个文件；下载后逐文件SHA及字节数核验通过。
manifest SHA256为`10ed1556accb8eaf46e5798cbc860e566583b303b38a5ea308bfb4a3d78ba0da`，
23项远端保留数据另列于manifest，不混同为已公开原始数据。

## 独立首差、持续触发与恢复

B1/B8的两次fault均从原始`dump/stack/construct`独立得到同一首差：
`prefill → model.layers.0.self_attn → Functional.scaled_dot_product_attention.0.forward → output Mean`。
该节点输入12项有限统计相同，第一层Q/K/V完整逻辑SHA也相同；首差不由host命中或期望profile指定。

| case | baseline输出Mean | 两次fault输出Mean |
| --- | ---: | ---: |
| B1 | -0.0019010795513167977 | -0.001901080016978085 |
| B8 | -0.0021081045269966125 | -0.0021081049926579 |

两次controlled均逐层切换M128/nRatio8至M64/nRatio4，28层prefill输出均有完整字节哈希变化。
decode没有注入；其第0层输出保持一致，后27层及logits承接prefill/cache差异。
B1目标请求及B8全部8个请求在prefill/decode均出现logits变化；重复fault稳定，
关闭后全部层、全部请求的输出SHA和完整私有logits字节均恢复到native。
每次原API只执行一次，输入before/after逻辑字节、身份及几何均保持；仅格式描述允许已记录的0→2规范化。

20个phase对照中，calibration/native的4个phase源栈不齐，保留不可对齐；其余16个对照源栈对齐。
12个calibration/native、fault重复及off恢复phase的完整输入/输出/logits哈希一致。
每个phase对照的122项不可观测字段保留，没有将null判为相等；本矩阵未见SDPA输出统计碰撞。
本地重新计算原始统计及560份调用记录通过，报告SHA为
`e520a201710f63b67b721982484a1af2f5c1c5fc4ecba26745a319d68ac3fb64`。

本次达到固定版本与形状下的**完整模型覆盖、原生策略切换、数值变化与模型传播验收**。
msprobe定位的是框架SDPA输出边界，未唯一识别kernel内部Reduce指令，也未据logits差异评估准确率或认定历史FIA缺陷复现。

本实验保留真实 Transformers Qwen3-0.6B FP16 的完整28层计算：

```text
真实请求 input_ids / position_ids
→ 原始 Qwen3Attention
→ 原始 F.scaled_dot_product_attention（每次恰好调用一次）
→ C++ SDPA backend → 原生 FlashAttentionScore
→ 真实输出投影、残差、后续层及 lm_head
```

没有替换 attention 算法，也不以替换 `torch_npu.npu_fusion_attention_v3` Python 名称证明内部拦截。
当前 FA host 构建和 `cann9.1-s1s2-v1` 证书保持不变；本轮新增模型边界和覆盖取证。

## 预注册范围

| 项目 | 本次固定环境 |
| --- | --- |
| 设备 | 124物理卡5，Ascend 910B4，单卡/rank0；保留卡2外部服务 |
| CANN | 9.1，`V100R001C11SPC001B243` |
| 模型/框架 | 原始 Qwen3-0.6B；PyTorch 2.10.0、torch-npu 2.10.0.post4、Transformers 5.5.4 |
| 模型执行 | FP16、原生 HF SDPA、2048-token prefill及一次固定token decode、真实KV cache |
| 注入 | `cann9.1-s1s2-v1`，`joint`，每次符合条件选择 M64/nRatio4 |
| 统计 | `statistics / mix / summary_mode=statistics`；已披露的FP32 high collector |

真实模型配置为28层、hidden size1024、Q heads16、KV heads8、head dimension128。
原始未量化 BF16 checkpoint 由原生 Transformers 加载为 FP16；本轮不等同于 vLLM FIA 服务接入。
模型、分词器、已安装 HF/SDPA 源码、运行时和私有 vendor 均逐文件固定。

冻结8个不同自然文本请求，各含2048个 prefill token 及1个固定后续 token。
B1执行 request_00；B8执行 request_00…07。每个 arm 的后续 token 相同，使用真实 KV cache，
避免生成结果分叉后输入也随之分叉；因此不是自由生成质量评估或 HTTP 服务压测。

每个case各自使用一个模型进程，在该进程内依次执行五轮：

```text
calibration → native baseline → controlled → controlled → off
```

每次运行创建新 cache，包含完整 prefill 和一次真实 decode，每个阶段均经过28层。
两组实际560次 SDPA，112次故障 prefill 策略切换、112次故障 decode 透传。
单独的 pilot 为 B1 calibration 一次 prefill+decode（实测56次），不计正式验收。
当前只选择已验收的 `joint` 候选，按 `every_match` 每次选择合法的 M/nRatio 替代策略。

证书只支持 B1/B8、FP16 BNSD、S1=S2=2048 的固定 FA 分支。
decode 的 Q长度1、KV长度2049明确超出该范围，必须记录为 unsupported 并验证未注入。
不能把它从模型调用总数删除，也不能把范围限制写成所有形状均支持。

## 上下文与全量覆盖证据

每层拥有独立 adapter/Session，保留本层 native 基线，避免几何相同导致跨层缓存覆盖。
calibration 的模板只用于进入不修改策略的 native 观察作用域，明确标为未验证意图。
只有真实 host 公共 IR19/attrs、完整 launch raw、205项解码及169项执行计划通过后，
才将该层观测作为基线。内部 causal mask 的真实描述符来自 native 公共 context，
不声称 Python 中存在或采集了该临时 mask 张量。

后续每次调用先核实际 SDPA Q/K/V、请求行映射和固定源码转换契约，再由 host 独立核实际 native 输入。
每个 eligible candidate 必须有匹配的原始 host 事件、公共 context 和实际 launch；
缺失不能判为“不满足条件”，必须作为 UNKNOWN 阻断全覆盖验收。

覆盖分母由三种来源交叉建立：实际模型配置及无条件模块 pre/post hooks、原始 msprobe
SDPA/construct 模块关系、整段 CANN 日志中的 attention launch census。
注入日志只提供实际策略的后验对应，不能决定哪些调用存在或应进入分母。
实际调用、eligible、unsupported、UNKNOWN、实际切换分别计数；任何遗漏或重复都不能通过。

B8一次原生调用覆盖8个请求槽。需同时报告原生调用覆盖率和 layer/request 映射覆盖率，
不能把28×8映射对说成224次独立 kernel 注入。请求级完整 logits 与中间逻辑哈希逐槽保存，
整批 msprobe 摘要不能拆成8份请求级误差。

## 数值与统计验收

沿用 `task=statistics / level=mix / summary_mode=statistics`，保留原始 dump、stack、construct、
配置与回执，不将整批 `.pt` 作为主交付。沿用并披露已校验的任务本地 FP32 high collector。
观察函数的诊断读回保留在原始数据中，分析按精确源码栈标记，不丢弃真实 SDPA 节点。

native/F/off共用同一个 `StrategySession.execute` 调用位置。calibration与之栈不同，
不能丢帧强造统计对齐；其透明性另外比较完整输出哈希。
每组B/F全模型对照只报告一个独立的首个可观测差异，后层输入变化可能来自上游传播，
不能将每层输出差都宣称为该层独立数值因果。每个原调用自身的输入 before/after 必须保持逻辑字节不变。
关闭后检查各层所有请求的输出及最终 logits 恢复，两次 controlled 检查持续命中和重复稳定。

运行只使用124空闲卡5，保留外部服务。控制器逐次门禁、共享资源锁、16GiB/8CPU限制，
并只清理带本任务身份的容器。校准使用900秒/case、1800秒总时限及2048MiB日志上限。
正式五轮按实际校准体量选择3600秒/case、7200秒总时限及16384MiB/case日志上限。
两组启动前至少留44GiB可用磁盘，运行和收尾至少留8GiB；预算和逐次检查随完成回执保存。
该调整保留完整DEBUG日志及字节校验，不改变输入、策略矩阵、23文件运行时或670文件vendor。
完整 logits 二进制留在124的 private_output；公开包仅包含统计、元数据、来源及审计。

## 采集实现和证据边界

最终校准`pilot_stream_a1`与正式矩阵使用完全相同的47个执行源码文件，校准准入报告SHA为
`37e73c33cf47b3023404d17d471ae3831d0219ecde0be6291a7efeaa2994e465`。
顺序reader保存原始调用窗口，用未修改的`actual_launch()`精确字符解析函数解码，累计完整前缀SHA、
行号和偏移；它不重哈希已消费字节。独立验收仍从完整原日志核对前缀及窗口并重解析raw，
最后一轮decode另由完整census覆盖。66个既有真实窗口的CPU逐字段/逐字节等价检查保留，不能代替实机验收。

早期`formal_a1`因重复扫描增长中的日志前缀而由本任务STOP停止，67个已返回调用及原始停止回执保留，
不计正式通过；最终正式B1/B8分别耗时388.84/514.13秒，包含采集和同步开销，不是kernel耗时。
两份完整DEBUG日志分别8653608395/8775055620字节，保留在124。
公开包保留原始statistics/mix三件套、560份调用、280份host/raw/POD记录及原始窗口。
完整logits、整段DEBUG日志和prompt/token原文不上传；因此仅凭公开子集不能重跑整日志漏调用检查、
全前缀验算或私有logits字节比较。manifest明确列出远端保留文件的路径、SHA与bytes。
