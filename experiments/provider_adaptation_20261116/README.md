# 5条 provider 与 Fault39 版本适配：124 推理验收

本轮完成部署实现、实际推理调用验证及 B/F/OFF 恢复。**统计可观测首差与原生机制结论分开记录**：
33 的统计首差在后续算子；NaN 或 null 不作为相等证据。所有用例均为推理前向，
没有 backward、optimizer 或训练故障注入。曾占卡的训练/ATK任务只是外部资源占用。

按用户要求，本分支只提交实现、版本调查和总结；不提交本轮原始 dump/stack/construct、
完整张量、模型权重或 vendor 二进制。历史机制和原有采集不覆盖。

## 推理范围与恢复

| 项目 | 本轮模型路径 | 实际覆盖与结果 |
|---|---|---|
| 20 | 完整 Qwen3-0.6B 与 SmolLM2-135M/Llama；BF16、B2单token前向 | 自动发现197/211个Linear，每组分别591/633次native调用；三轮重复稳定、输入保持、OFF精确恢复。B/F权重均显式转换NZ29；不继承旧INT2/INT4验收 |
| 26 QMM | 真实 DeepSeek-R1-Distill-Qwen-32B-W8A8，vLLM eager TP2；8请求、prefill+decode、生成2token | 每rank45次符合条件的调用全部命中、消费并恢复，待恢复数为0；OFF精确恢复 |
| 26 FIA | 同一真实32B W8A8推理链，未人为增加producer或consumer | 每rank126次有效写入及实际消费者恢复；64个Attention候选中首层没有当前forward的AddRmsNormBias producer，明确透传；4/8请求token及文本变化，OFF精确恢复 |
| 共享量化尺度 | 真实32B W8A8、CANN9.1、TP2/B8 pure decode | 自动选第一个实际eligible量化模块；每rank3次DynamicQuantV2命中；实际scheduler行/前缀/健康输入绑定；6/8请求logprob变化、token不变；重复稳定、OFF精确恢复。不是全部层注入 |
| 33 | 完整Qwen3、16383token prefill后固定teacher decode，B8/Q1/KV16384 | 共84次实际SInner8192→1024（28层×3轮，每次含8请求）；完整page逆映射与launch审计通过；logits maxabs=0.02734375，Top1不变；OFF精确恢复。三次是同一decode重复，非连续生成质量验收 |

26 QMM 的8条请求中，1条logprob变化，token及文本均不变。

## 注入边界与调用链

- 20：模型结构发现 → 实际Linear forward → CANN cache Run的typed formal/slot/storage核对 → 原v17临时输入mantissa变换 → 原算子计算。
- 26 QMM：当前forward的实际AddRmsNormBias残差 → allocator两个active相邻block → 原生QMM output-base underflow → 恢复正常output/padding → 真实consumer消费受损残差 → 恢复残差。
- 26 FIA：当前forward的实际残差 → 原生FIA V3的输出destination改绑 → 保持正常FIA返回内容 → 真实AddRmsNormBias读取被覆盖残差 → 恢复。首个数值差异应在残差读取/消费处，而非要求FIA正常返回值立即变错。
- 共享量化：结构与实际健康调用选点 → scheduler请求/行/前缀一致 → 原DynamicQuantV2逐行归约切换为batch共享范围 → 后续量化计算。原算术保留，新增TLS选择与owner线程约束。
- 33：实际AttentionInterface及存活DynamicCache → 全值page-table逆映射核验 → 原生paged IFA host合法SInner分组切换 → 实际launch与数值观察。不是dense prefill FIA。

实现入口见[部署README](../../fault_injection/deployments/README.md)、
[buffer20.py](../../fault_injection/deployments/buffer20.py)、
[residual26.py](../../fault_injection/deployments/residual26.py)、
[batch_scale.py](../../fault_injection/deployments/batch_scale.py)、
[paged33.py](../../fault_injection/deployments/paged33.py)。
用户不提供层名；部署人员仍须准备匹配版本、库指纹和独立证据目录。

## 独立 msprobe 首差

正式两组使用原生 `task=statistics / level=mix / summary_mode=statistics`，完整API/module采集，
没有算子过滤。比较只读两组原始三件套、配置和采集器身份，不读native命中、策略日志或完整数组。
公式固定为 `abs(F-B) > 1e-8 + 1e-5*abs(B)`，不降阈值。以下数值是统计字段，不是逐元素差值。

| 组/Rank | 实际step | 独立结果/记录 | 字段：B → F | 首差前/处不可观测数 |
|---|---|---|---|---|
| provider20-model-a1 | 0 | `Functional.linear.0.forward/output/0` | `/Max`: `3.4375` → `2.296875` | 108 |
| provider20-model-a1 | 1 | `Functional.linear.0.forward/output/0` | `/Max`: `3.4375` → `2.296875` | 108 |
| provider20-model-a1 | 2 | `Functional.linear.0.forward/output/0` | `/Max`: `3.4375` → `2.296875` | 108 |
| provider20-smollm-a1 | 0 | `Functional.linear.0.forward/output/0` | `/Max`: `4.125` → `2.9375` | 108 |
| provider20-smollm-a1 | 1 | `Functional.linear.0.forward/output/0` | `/Max`: `4.125` → `2.9375` | 108 |
| provider20-smollm-a1 | 2 | `Functional.linear.0.forward/output/0` | `/Max`: `4.125` → `2.9375` | 108 |
| provider26-final-a2 / FIA / rank0 | 0 | `_C_ascend.npu_add_rms_norm_bias.2.forward/input_args/1` | `/Max`: `23.5` → `23.25` | 552 |
| provider26-final-a2 / FIA / rank0 | 1 | `Torch.index_select.0.forward/output/0` | `/Max`: `15920` → `149742` | 6 |
| provider26-final-a2 / FIA / rank1 | 0 | `_C_ascend.npu_add_rms_norm_bias.2.forward/input_args/1` | `/Max`: `23.5` → `23.25` | 552 |
| provider26-final-a2 / FIA / rank1 | 1 | `Torch.index_select.0.forward/output/0` | `/Max`: `15920` → `149742` | 6 |
| provider26-final-a2 / QMM / rank0 | 0 | `Triton._triton_rope.7.forward/input_args/0` | `/Max`: `7.5` → `7.46875` | 2064 |
| provider26-final-a2 / QMM / rank0 | 1 | `NPU.npu_dynamic_quant.18.forward/output/1` | `/Min`: `0.009904035367071629` → `0.00996555108577013` | 1381 |
| provider26-final-a2 / QMM / rank1 | 0 | `不可比较：NONFINITE_UNCOMPARABLE` | `/Max`: `NaN` → `NaN` | 2654 |
| provider26-final-a2 / QMM / rank1 | 1 | `NPU.npu_quant_matmul.17.forward/output/0` | `/Max`: `1.4765625` → `1.4921875` | 1330 |
| quant-model-a2 / rank0 | 0 | `NPU.npu_dynamic_quant.0.forward/output/0` | `/Max`: `127` → `98` | 160 |
| quant-model-a2 / rank0 | 1 | `NPU.npu_dynamic_quant.0.forward/output/0` | `/Max`: `127` → `98` | 160 |
| quant-model-a2 / rank0 | 2 | `NPU.npu_dynamic_quant.0.forward/output/0` | `/Max`: `127` → `98` | 160 |
| quant-model-a2 / rank1 | 0 | `NPU.npu_dynamic_quant.0.forward/output/0` | `/Max`: `127` → `98` | 160 |
| quant-model-a2 / rank1 | 1 | `NPU.npu_dynamic_quant.0.forward/output/0` | `/Max`: `127` → `98` | 160 |
| quant-model-a2 / rank1 | 2 | `NPU.npu_dynamic_quant.0.forward/output/0` | `/Max`: `127` → `98` | 160 |
| provider33-model-a2 | 0 | `Tensor.mean.3.forward/output/0` | `/Max`: `0.053691256791353226` → `0.05369321256875992` | 328 |
| provider33-model-a2 | 1 | `Tensor.mean.3.forward/output/0` | `/Max`: `0.053691256791353226` → `0.05369321256875992` | 328 |
| provider33-model-a2 | 2 | `Tensor.mean.3.forward/output/0` | `/Max`: `0.053691256791353226` → `0.05369321256875992` | 328 |
| model-851-bf-a1 | 0 | `NO_OBSERVABLE_STATISTICS_DIFFERENCE` | — | 6102 |
| model-851-bf-a1 | 1 | `Functional.linear.3.forward/output/0` | `/Max`: `0.76953125` → `+Inf` | 258 |
| tp2-851-bf-a1 / rank0 | 0 | `NO_OBSERVABLE_STATISTICS_DIFFERENCE` | — | 6372 |
| tp2-851-bf-a1 / rank0 | 1 | `Functional.linear.3.forward/output/0` | `/Max`: `0.625` → `+Inf` | 302 |
| tp2-851-bf-a1 / rank1 | 0 | `NO_OBSERVABLE_STATISTICS_DIFFERENCE` | — | 6372 |
| tp2-851-bf-a1 / rank1 | 1 | `Functional.linear.3.forward/output/0` | `/Max`: `0.369140625` → `+Inf` | 302 |
| model-91-bf-a2 / rank0 | 0 | `NO_OBSERVABLE_STATISTICS_DIFFERENCE` | — | 6426 |
| model-91-bf-a2 / rank0 | 1 | `Functional.linear.3.forward/output/0` | `/Max`: `0.625` → `+Inf` | 366 |
| model-91-bf-a2 / rank1 | 0 | `NO_OBSERVABLE_STATISTICS_DIFFERENCE` | — | 6426 |
| model-91-bf-a2 / rank1 | 1 | `Functional.linear.3.forward/output/0` | `/Max`: `0.369140625` → `+Inf` | 366 |

26 FIA的step0首差在consumer的残差输入，符合写目标错误的传播位置。step1已经使用不同的生成token，
因此首先在token读取处变化；不能把该step当作同输入的新根因定位。
26 QMM rank1的step0被双方NaN阻断，整体结论为`EARLIEST_DIFFERENCE_UNPROVEN`；step1的可观测差异不能补足此前不可比较的记录。
33在固定容差下只能独立看到后续mean，不能仅凭统计唯一归因到IFA。统计Max/Min/Mean/Norm是整张量汇总，
既不是逐请求统计，也不能证明所有此前隐藏值相同；native证据用于另行核验机制，不倒推或指定统计首差。

三轮固定前向/teacher decode的数据是step0–2；26真实生成只采到prefill/decode的step0–1；
Fault39算子级是step0，模型是实际step0–1。没有补造或重命名step。

## Fault39：按实际接口和构建适配8.x/9.x

四个实际安装8.5.1、9.0.0、9.1.0、9.2.0的新构建均通过算子发现、self/mat2/dual lossless、
真实有限输入替换、原输入storage保持和B/F/OFF；四版本原始统计首差均为
`Torch.matmul.0.forward/output/0` Max `-0.3359375 → +Inf`。
8.5.1另完成Qwen3 TP1/TP2真实推理；9.1真实TP2推理结果见下表。9.0/9.2仅取得本轮算子级结论。

| 模型回执 | 状态 | 所选rank | 通过阶段/恢复 |
|---|---|---|---|
| `model-851-lossless-a1/model_lossless/lossless/validation.json` | LOSSLESS_VALIDATED | [0] | [(True, True), (True, True), (True, True)] / cleanup=DISABLED_CONFIRMED |
| `model-851-bf-a1/model_baseline/validation/validation.json` | BASELINE_MODEL_CAPTURED | [0] | [(True, True)] / cleanup=DISABLED_CONFIRMED |
| `model-851-bf-a1/model_fault/validation/validation.json` | PATTERN_NATIVE_VALIDATED_MSPROBE_ANALYSIS_PENDING | [0] | [(True, True)] / cleanup=DISABLED_CONFIRMED |
| `tp2-851-lossless-a1/model_lossless/lossless/validation.json` | LOSSLESS_VALIDATED | [0, 1] | [(True, True), (True, True), (True, True)] / cleanup=DISABLED_CONFIRMED |
| `tp2-851-bf-a1/model_baseline/validation/validation.json` | BASELINE_MODEL_CAPTURED | [0, 1] | [(True, True)] / cleanup=DISABLED_CONFIRMED |
| `tp2-851-bf-a1/model_fault/validation/validation.json` | PATTERN_NATIVE_VALIDATED_MSPROBE_ANALYSIS_PENDING | [0, 1] | [(True, True)] / cleanup=DISABLED_CONFIRMED |
| `model-91-lossless-a2/model_lossless/lossless/validation.json` | LOSSLESS_VALIDATED | [0, 1] | [(True, True), (True, True), (True, True)] / cleanup=DISABLED_CONFIRMED |
| `model-91-bf-a2/model_baseline/validation/validation.json` | BASELINE_MODEL_CAPTURED | [0, 1] | [(True, True)] / cleanup=DISABLED_CONFIRMED |
| `model-91-bf-a2/model_fault/validation/validation.json` | PATTERN_NATIVE_VALIDATED_MSPROBE_ANALYSIS_PENDING | [0, 1] | [(True, True)] / cleanup=DISABLED_CONFIRMED |

上表保留原生采集回执的原始状态。`MSPROBE_ANALYSIS_PENDING`表示回执生成时尚未运行比较；本轮后续独立比较已完成，结果列在上一节，未改写原回执。

版本名不再作为8.5.1路径白名单；真实头文件探针、必需符号、库文件及加载可执行段SHA、
私有SONAME代理、匹配set_env和真实lossless回执共同约束具体构建。未安装的8.0–8.4等版本
没有实测证书。公共签名相同不等于私有ABI/tiling语义兼容。完整差异见
[CANN版本适配说明](../../fault39/docs/cann_version_adaptation.md)。
Fault39仍是有限BF16正负抵消输入经过真实Matmul产生范围异常，没有事后强写Inf；
不声称已定位kernel内accumulator类型。统一目录39桥和黄区内源框架接线仍单独待适配。

## 修复、失败保留与检查

- 26 FIA初次真实推理失败揭示跨forward残差复用。修正后B/F全部重采，且QMM共享生命周期也重新验收；未将旧构造用例代替模型通过。
- QMM构造小用例因实际block不相邻拒绝，保留；实际模型用真实allocator关系验证，没有放宽邻接检查。
- 共享量化旧process-only后端被多线程worker门禁拒绝；新增并实际测试TLS选择/执行context owner约束，没有绕过原线程门禁。
- Fault39早期加载DSO解析错误、9.0混合set_env、9.1 vLLM默认KV block16不兼容均保留；9.1使用实际支持的block128重新运行。
- 33原生审计独立匹配call_id/pid/tid/时间窗、最终typed/raw及actual launch。未声称读到kernel参数raw；enter日志缺scope_token字段也明确保留。
- 本地181项统一接口/生命周期、69项FA策略回归、Fault39 54项核心合同及47项工具回归通过；原生TLS foreign-thread检查另行通过。CPU结果不算NPU验收。
- 原始msprobe在部分BF16/私有格式统计中有null；不补零。双方NaN会阻断首差确定，不报“相等”。

## 复现实测入口

公开case JSON保留本轮124的实际镜像、只读挂载和入口，属于冻结实验配置；新的部署应使用新源码/构建/attempt目录，不覆盖原件。
20使用`provider20_model_cases.json`与`provider20_smollm_cases.json`；26使用修复后重采的`provider26_final_cases.json`；
共享量化使用`quant_cases_a2.json`；33使用`provider33_cases_a2.json`。这些入口由`provider_run_124.py`串行执行，每批最多8例，逐例检查资源锁与设备空闲。
Fault39算子矩阵由`run_fault39_matrix.py`调用其独立controller；模型先运行对应lossless case，再将真实通过回执提供给B/F case。
不能跳过lossless门槛，也不能将历史路径中的库复制到其他CANN安装后直接使用。依赖指纹与构建方式见[部署README](../../fault_injection/deployments/README.md)。

## 证据与复核

[validation_summary.json](validation_summary.json)包含结果标志、覆盖计数、独立首差与原件SHA，
[source_review.json](source_review.json)记录待提交源码与实际部署来源核验，
[checksums.json](checksums.json)校验本次公开文件；`python experiments/provider_adaptation_20261116/verify_publication.py`可核对。SHA按LF规范化，以兼容Git在Windows的换行转换。
实际原件保留在124的 `/home/provider-adaptation-20261116/`；最终26为`provider26-final-a2`，
其他最终目录见summary。`compare_pair.py`调用同一原始比较器，可分别指定`--rank 0/1`复算。
GitHub只含总结，外部agent仅下载此分支无法重算完整原始首差，须取得所列私有采集目录。
新的原始statistics不在Git提交范围，也不上传新ZIP。

本轮使用专用容器、共享资源锁、每例空闲门禁；未恢复过期Docker入口隔断。
发布基线为`052db5c1db1ab9ea6b85eb9ce5020d8a95e3ac4d`，分支`codex/native-provider-adaptation`。
