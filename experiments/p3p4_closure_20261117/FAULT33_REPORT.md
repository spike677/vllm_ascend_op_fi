# Fault33：真实连续生成、分页证据与恢复

本轮使用124物理卡5、910B4/A2、CANN9.1.0、FP16、预训练Qwen3-0.6B，真实HF eager推理和分页 `npu_incre_flash_attention`。这是独立HF适配验收，**没有取得公共vLLM Worker Provider或任意模型/形状资格**。安装包与量化测试共用 `packages-p4-final-a1`，33使用自己的CANN9.1镜像和既有审计vendor。

**真实连续生成和OFF恢复通过；统计能检测差异，但首差位于下游，未取得“仅靠statistics唯一定位到IFA”的资格。** F的4步完整logits都变化，8请求的生成token和文本均未变。结果不等于错误答案或业务准确率下降。

## 固定实验与自然token反馈

8条自然文本种子分别重复并经固定tokenizer截取为16383 tokens。正常SDPA prefill返回真实KV缓存和首token，随后连续4次decode，每次输入上一步实际argmax token。没有重复teacher token、裁剪cache或替换生成结果。

| 实际decode step | 调用前 / 后KV长度 | 故障条件及策略 |
|---|---|---|
| 0 | 16383 / 16384 | B维持原策略，F在28层真实适用调用切换 SInner；已有审计几何不变 |
| 1 | 16384 / 16385 | 原生正常分页IFA，不扩展故障适用长度 |
| 2 | 16385 / 16386 | 原生正常分页IFA |
| 3 | 16386 / 16387 | 原生正常分页IFA |

仅原有 `B8/Q1/KV16384/H16/KV8/D128/page128` 条件允许注入；原生host重算对应参数，SInner由8192切为1024。page blockSize仍为128。本实验是同类切分策略的受控切换，不宣称复刻历史blockSize变化的全部触发条件，也不以二进制差异认定违反算子精度标准。

每个B/F进程独立执行 reference → comparison → OFF 三条轨迹；每条先建立新的prefill cache，实际生成5个新token（prefill首token＋4个decode输出）。正式 statistics/mix 只采 comparison 轨迹的完整4个decode，`step0–3`就是实际连续decode步数。reference/OFF是运行时恢复控制，不增加公开采集组。

## 分页、存储与覆盖

自动从实际模型的共享Attention注册调用发现28层，用户不填写层名。模块对象和原始调用源在msprobe包装forward前绑定，资格和scope仍在实际调用时核对。首次合格decode使用原NativeSession完整覆盖门禁，结束后关闭该故障scope；后3步使用单独的正常Attention路由，不能用它们推高故障覆盖率。

对每层真实KV，与当前DynamicCache的存储、offset、shape、stride和层身份绑定。页表逐请求独立、不交叉；逐元素逆变换核对全部有效KV。后续非整页长度分配129页/请求，尾部补零，实际长度始终传16385/16386/16387，补零区域不作为有效token。缓存对象保持同一身份，长度每步恰好增加1；Q/K/V页及页表在原生调用前后完整核验。

旧receipt中的键 `teacher_tokens` 在本实验保存的是实际前序argmax token，不代表teacher forcing。真正的输入token、输出token、KV前后长度与生成文本在独立运行时摘要中逐步保留。第一层B/F的实际Q/K/V和页表另行精确比对，不用它们决定统计首差。

## 统计与原生核验分离

原生msprobe采所有可支持的API/module输入输出，`statistics/mix`、`summary_mode=statistics`，无算子过滤。每步可见28条真实 `NPU.npu_incre_flash_attention` 记录，输入和输出保留实际shape；Mean/Norm为null，只能据可观测字段比较。原始dump/stack/construct、配置和采集器身份全部留在124。

独立比较器未修改，固定 `abs(F-B) > 1e-8 + 1e-5*abs(B)`，不读取native命中、host策略、私有完整张量或运行时结果。null、NaN及无法对齐的记录均保留。先结束原始统计判定，再独立核验完整进程日志的NNOP launch、SDK BMM、公共host typed POD和最终序列化raw；不使用期望profile代替实际launch。

## 正式结果

正式attempt为 `continuous33-a3`，B/F各exit0。两进程的正常reference轨迹完整logits逐步一致。每个进程的comparison和OFF都从新的prefill开始；B comparison及两组OFF的4步完整logits精确等于其reference，cache同一对象逐步增长、页映射完整有效、原输入未被原生调用改写。

完整进程的原生审计通过：B/F各**336个实际IFA launch**，每组84次原已合格几何的host/POD/raw核验，以及252次后续正常长度的launch核验。F comparison的28层实际SInner均为1024，reference/OFF均为8192；B各阶段均为8192。没有未归属、重复或无法解析的同进程IFA调用。见 [B原生审计](summaries/continuous33-baseline-native.json)、[F原生审计](summaries/continuous33-fault-native.json)及[执行与构建来源](summaries/continuous33-execution-manifest.json)。

| decode step | F相对reference的全词表最大绝对差（8请求取最大） | 生成token变化 | OFF完整logits |
|---|---:|---|---|
| 0 | 0.02734375 | 0/8 | 精确恢复 |
| 1 | 0.03515625 | 0/8 | 精确恢复 |
| 2 | 0.0390625 | 0/8 | 精确恢复 |
| 3 | 0.06640625 | 0/8 | 精确恢复 |

上述最大差是完整数组的辅助逐元素核验，**不是原始统计首差的输入，也不是相对百分比**。最后一步最大差发生在索引5的请求（第6条）。逐请求差值、实际生成token/text、KV长度和恢复摘要见 [连续生成结果](summaries/continuous33-generation.json)。第一层B/F的真实Q/K/V、页表、实际token及几何精确一致；后续变化不能重新解释为新的故障命中。

| step | 独立首个可观测差异 / 实际模块 | 字段及原始 B→F | index / 之前null |
|---|---|---|---|
| 0 | `Tensor.mean.3.forward/output/0`；层0 `post_attention_layernorm` | Max **0.053691256791353226→0.05369321256875992** | 164 / 328 |
| 1 | `Tensor.mean.7.forward/output/0`；层1 `post_attention_layernorm` | Max **0.09827947616577148→0.09828834235668182** | 490 / 980 |
| 2 | `Tensor.__mul__.26.forward/output/0`；层1 `post_attention_layernorm` | Max **32.09375→32.125** | 500 / 1000 |
| 3 | `Functional.linear.10.forward/output/0`；层1 `self_attn.o_proj` | Max **0.6728515625→0.67333984375** | 475 / 950 |

这里的 `Tensor.mean` 是模型RMSNorm实际执行的算子，其输出的Max被采集；采集统计字段Mean仍为null。模块归属仅由原始stack/construct唯一匹配获得，B/F来源一致，见 [首差上下文](summaries/continuous33-first-context.json) 和 [解析器](summarize_stat_context.py)。

B自比四步均为 `NO_OBSERVABLE_STATISTICS_DIFFERENCE`：step0比较5368项记录、10736项null；step1–3各7272项记录、14544项null。调用结构对齐，未遇到NaN阻塞，但不能把null解释为数值相等。逐步原始统计摘要见 `summaries/continuous33-statistics-step0..3.json`，保留原始三件套、配置、采集器与私有完整报告哈希。

**统计结论是下游数值差异及传播，不是IFA唯一根因定位。** 最早可观测位置与真实注入边界必须分别记录；正常策略的后续decode仍可带着已变化的KV继续产生logits差异。

## 失败保留与限制

- `continuous33-a1`的参考轨迹完成4步，但旧覆盖检查把后3步不合格正常调用计为未观察，按原门禁失败。新实现分离两个scope，没有放宽原Native覆盖要求。
- `continuous33-a2`完成B参考及4步统计，comparison完整logits均精确恢复；进入OFF时源代码发现被采集器forward包装遮住，因此失败，F未运行。三阶段原始模块绑定已提前到采集之前。旧B自比有10736/14544项null但无可观测差异；不能把它记为完整B/F/OFF通过。见[原失败摘要](summaries/continuous33-prior-a2-summary.json)。
- 当前claim限定于真实HF轨迹和审计vendor；后续正常长度不是新的故障策略资格，公共`all_eligible`仍关闭。没有评估任务准确率、任意长生成或默认SDPA与分页适配的全量精度等价。
- 公共host的最终raw与实际launch参数核对，不等于直接抓到了kernel参数区的raw字节；报告保留 `kernel_argument_raw_observed=false`。

实现：[连续分页适配](../../fault_injection/deployments/paged33_continuous.py)、[实际生成脚本](probe33_continuous.py)、[独立统计比较](compare_pair_124.py)、[完整launch核验](verify33_continuous_launch.py)、[运行时摘要](summarize33_continuous.py)。新原始统计、完整张量及vendor二进制不提交GitHub。

124原件在 `/home/p3p4-closure-20261117/continuous33-a3/{continuous33-baseline,continuous33-fault}`；原始统计为各case的 `serving/capture/observation/step0–3/rank0`。独立报告为 `continuous33-independent-step{0..3}-a3`。旧attempt a1/a2保留，未覆盖或删除。
