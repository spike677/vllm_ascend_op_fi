# 17 条机制：本轮实现、实测与未闭合项

**17/17 完成源码和门禁审计；新增 3 条构造 NPU 模块的 B/F/OFF 验收。本轮新增完整预训练模型故障闭环为 0，共享 Worker 实际故障闭环为 0。**
不能把 CPU 模型适配、真实 Worker 的只读发现或历史模型 PASS 填成新的完整闭环。

| 本轮结果 | 数量与范围 |
|---|---|
| 机制能力卡、具体实施方案及 G0–G9 缺口 | 17 |
| CPU 回归 | 445 项，0 失败、0 跳过；不是每机制各445项 |
| 实际模型结构与透明前向 | Qwen3 dense/MoE、GLM、DeepSeek V2/V3 MLA+MoE、Llama，共6种；CPU随机初始化小模型 |
| 新 NPU 机制级 B/F/OFF | 15、17、18；均为构造的 eval/inference_mode 模块 |
| 新统计独立全局首差通过 | 0/3；三个 step 均保留结构不齐 |
| 真实共用 Worker 负门禁与恢复 | Fault20、Fault39，Qwen3-0.6B TP1 |
| 真实 QMM 只读发现 | 32B W8A8 TP2，两 rank 各3次，仅第一轮 |
| CANN 精确环境 | 8.5.1、9.0.0、9.1.0、9.2.0 的镜像安装/ELF复核；历史 NPU 范围单列 |
| 硬件 | 本轮实际910B4/A2；A3未验收 |

## 17 条逐项结果

“本轮层级”不继承历史实测。L2包括设计合同/拒绝测试，不等于 native 已实现；L3在本表仅指实际 NPU 机制级。

| 机制 | 本轮层级与操作 | 当前阻塞/下一步 |
|---|---|---|
| [8.restore_offset](overnight_fault_cards/8.restore_offset.md) | L0，找回并校验历史配对 routing 源码 | 缺依赖与可信请求合同；要证明 indices/probs配对及真实MoE消费者 |
| [15.alias_detach](overnight_fault_cards/15.alias_detach.md) | L3，格式29视图、copy_、原base消费 B/F/OFF | 私有格式统计null；自然预训练模型写入链未证 |
| [17.spatial_padding](overnight_fault_cards/17.spatial_padding.md) | L3，真实Conv3d参数变化和OFF | 仅构造模块，预训练视觉路径未验收；B/F非张量padding参数不同 |
| [18.api_half](overnight_fault_cards/18.api_half.md) | L3，标准RMSNorm两实例8/8调用减半 | 自定义RMSNorm不放行；采集hook在返回代理前，不能说模块原始统计已减半 |
| [20.buffer_substitute](overnight_fault_cards/20.buffer_substitute.md) | L2及真实Worker只读发现/负门禁 | WeightNz零观测；QMM跨轮观测不稳定、无合格profile/replay，未注入 |
| [22.wrong_row](overnight_fault_cards/22.wrong_row.md) | L0，历史IndexSelect返回错误行语义审计 | 实现本体与cache消费者绑定缺失，不冒称索引溢出 |
| [25.reduce_scatter](overnight_fault_cards/25.reduce_scatter.md) | L2，通信组/同输入差值采集回放设计与合同 | 无真实collective adapter和group证据；CPU模拟不授予NPU资格 |
| [26.residual_stomp](overnight_fault_cards/26.residual_stomp.md) | L2，回归及历史写前NaN定位复核 | allocation初始化来源尚未闭合；rank1严格首差继续阻塞 |
| [26.fia_destination](overnight_fault_cards/26.fia_destination.md) | L2，保留历史真实推理writer/consumer链 | 共用Worker factory与可信回放未接入 |
| [quant.batch_shared_scale](overnight_fault_cards/quant.batch_shared_scale.md) | L2，保留历史TP2 decode及请求行映射 | 全张量统计不能证明每请求scale/Q变化，需逐行辅助证据 |
| [28.fa.legal_tiling](overnight_fault_cards/28.fa.legal_tiling.md) | L2，策略/资格CPU回归 | 原生FA调用与共用模型forward桥仍缺 |
| [28.fia.legal_tiling](overnight_fault_cards/28.fia.legal_tiling.md) | L2，保留合法切分历史对照 | 新router NPU未验收；合法切分不保证统计差异达标 |
| [28.fia.denominator_replay](overnight_fault_cards/28.fia.denominator_replay.md) | L2，保留独立分母累计错误及历史证据 | 匹配vendor构建和新router实机资格待补，不与合法切分合并 |
| [33.paged_decode](overnight_fault_cards/33.paged_decode.md) | L2，保留真实分页decode历史范围 | 固定阈值首差不能唯一定位IFA；连续生成、共享Worker尚缺 |
| [39.matmul_range](overnight_fault_cards/39.matmul_range.md) | L2及真实Worker隔离/恢复 | 同步七阶段回放需真实scheduler请求身份，缺host接口，未注入 |
| [39.output_proxy](overnight_fault_cards/39.output_proxy.md) | L0，历史native输出代理单独审计 | 专属Host合同与存储证明缺失，不继承range资格 |
| [41.tail_column](overnight_fault_cards/41.tail_column.md) | L0，dispatcher/cache写边界设计与拒绝 | 尚无真实dispatcher及尾列消费者证据，不伪造Provider |

完整环境、API、B/F、OFF、覆盖、下一步字段见[机器矩阵](../experiments/overnight_20261116/fault_matrix.json)。
基线历史级别、原文件SHA和当前结果分别保存在能力卡，旧实验不被本轮覆盖。

## 新增 NPU 机制验证

运行在124物理卡5、Ascend910B4、CANN8.5.1、vLLM-Ascend0.18.0镜像，torch-npu `2.9.0.post1+gitee7ba04`。
每组先正常重复，再预热一次、采三次同输入forward，最后退出session并比较OFF。
`step0–2`是采集迭代，不是生成token。全部组输入不变、输出有限、重复精确稳定、OFF精确恢复。

| 机制 | 精确运行检查（辅助证据，未参与统计首差选择） | 覆盖 |
|---|---|---|
| 15 | B更新64个指定元素且非目标不变；F视图写入-32成功，但原base保留旧值；同一base的`+1`结果正确读到旧值 | 1个构造模块，预热+3次均触发；MaxAbs=168 |
| 17 | 实际Conv3d padding `[0,0,0]→[0,7,7]`，输入和权重固定，输出shape仍相同而数值改变 | 1个Conv3d，4/4调用；消费者MaxAbs=0.4982621669769287 |
| 18 | 两个标准`torch.nn.RMSNorm`，每次返回乘0.5，下游相加与健康输出减半精确一致 | 2模块×4 forward，8/8调用；消费者MaxAbs=1.7051955461502075 |

15的OFF使用重新建立的正常base，不意味着关故障自动修复已损坏base。17尚未验证完整独立CPU卷积参考。
18是API返回值代理，不能称为CANN Reduce内部精度错误。这里的重复覆盖只授予这三个构造模块，不授予共享Worker全模型多目标资格。
[完整摘要与原件哈希](../experiments/overnight_20261116/mechanism_receipts-a1.json)

## 原始 statistics/mix 独立结论

B/F同配置，`task=statistics`、`level=mix`、`summary_mode=statistics`、`precision=high`，没有算子过滤。
原生三件套、config、collector身份和完成文件均保留在124。
比较器仅读这些文件，使用固定`rtol=1e-5, atol=1e-8`，不读命中日志或完整数组。

**三条机制的九个step均为`ALIGNMENT_MISMATCH`，组级为`INCOMPLETE_COMPARISON`；无可声明的全局独立首差。**

| 机制 | 原始采集观察到的局部现象 | 为什么不授予全局首差 |
|---|---|---|
| 15 | F新增格式转换；检查API的布尔结果和统计改变；原base消费者输入/输出Max/Min/Mean/Norm均null | null不可观测，新增调用使整体调用链不齐；检查API变化不能替代消费者定位 |
| 17 | 同栈Conv3d模块输出Max `0.5635518431663513→0.24107059836387634`（57.223%）；Min `-0.4876461327075958→-0.39287781715393066` | conv3d的非张量padding参数改变；现有严格比较器拒绝全结构对齐 |
| 18 | 原RMSNorm模块统计相同；F新增两个乘0.5的API；相同下游add的Max `3.410391092300415→1.7051955461502075`（50%） | 模块hook观察发生在返回代理前；新增API不能忽略后宣称全局首差 |

局部比较亦只使用原始统计；它是诊断结果，不覆盖严格验收失败。
三个step现象相同。对齐失败时旧摘要中`unobservable_count=0`表示未进入数值遍历，**不表示没有null**。
详见[独立诊断、每步字段及哈希](../experiments/overnight_20261116/STATISTICS_ALIGNMENT_DIAGNOSIS.md)。

## 共用 Worker：实际通过与未通过

正常vLLM创建的临时Worker加载同一扩展。Fault20、Fault39分别检查匹配preload，然后注册因缺少可信回放被拒绝。
未恢复时同Worker其他Provider也被拒绝；owner解除注册、detach后隔离解除，正常单token生成与前测精确相同。
这证明恢复门禁，**不是注入后OFF验收**。隔离只约束扩展的注入RPC；普通serving隔离、客户端超时及SIGKILL由host控制。

WeightNz观察两轮、113个候选、226次模块调用，原生事件0。
QMM在真实32B W8A8 TP2推理中，两rank各观察3个native调用，均只在forward0；forward1为0。
每rank 258结构候选、514次模块调用中511次未记录到native事件。
状态为`PROFILE_REQUIRED`，不能称为三个稳定目标、全量覆盖或自动选点注入完成。
该32B检查点实际架构是Qwen，不算DeepSeek架构的NPU验收。

F20仍缺真实上下文捕获、完整输出/输入存储与排他回放资格。
F39的七阶段同步回放逐次要求新真实scheduler请求ID；普通driver不能在该Worker被同步RPC占用时提供下一轮调度。
本轮未伪造请求ID，也未减弱unique/单次门禁。
[实际回执](../experiments/overnight_20261116/worker_receipts-a2.json) ·
[代码位置、原件hash及最小接线设计](../experiments/overnight_20261116/worker_injection_gap.json)

## 模型、版本及包部署范围

自动适配依据实际class/module/config及维度，补充DeepSeek MLA低秩投影角色。
六种CPU模型完成透明prefill、cache decode和OFF，但不把随机小模型等同预训练模型质量验收。
[模型矩阵](model_adaptation_matrix.md)

对4个CANN镜像的16个库复核SHA、SONAME和依赖；成功读取的ELF notes无GNU BuildID则保留null。
8.0/8.1/8.2/8.3/其他RC及A3没有本轮实物验收，不按8.x/9.x大版本放行。
已有8.5.1/9.0/9.1/9.2 Fault39原生证据保留，不能转授其他Provider。
[环境矩阵](cann_hardware_compatibility.md)

三个纯Python包在Windows隔离目录和124 aarch64无设备容器安装、独立导入通过；包内无vendor二进制、权重或原始dump。
本轮真实Worker按冻结源码`PYTHONPATH`加载，尚未做“新wheel安装后Worker注入”验收。
[本地包回执](../experiments/overnight_20261116/package_receipt.json) ·
[124安装回执](../experiments/overnight_20261116/package_124_receipt.json)

## 失败、恢复及证据保留

- 18首次F在ACL初始化被设备占用挡住，后续门禁也观测到外部任务；等待自然释放后新attempt成功，没有停止外部任务。
- 17首次B因采集器后装覆盖conv3d包装器，清理拒绝；F未启动。将采集器提前到session安装前后，新B/F通过，未放宽owner清理门禁。
- Worker首次可调用对象RPC因序列化配置失败；仅独立临时容器启用可信本地函数序列化后完成只读观察。生产扩展RPC仍为具名方法，未改服务安全配置。
- C++ CPU初次构建被既存格式缩进warning作为error阻塞；重试仅取消该warning的error升级，保持其他断言与检查，native源码未改。

历史26的NaN复核发现FIA输入链前的目标赋值对象写前NaN，右值及写后Attention输出有限；未证明其分配初始化来源。
因此未删除NaN记录，原严格首差阻塞不变。[分析](../experiments/overnight_20261116/fault26_nan_analysis.json)

原始采集仍在 `/home/overnight-17-20261116`，发布的回执列出逐文件SHA。
历史31/85文件和旧Native构建作为旧提交/原件保留，新分支不回写旧快照；源码冻结a1/a2/a3分别记录实际运行版本。
提交范围及校验见[实验目录](../experiments/overnight_20261116/README.md)。
