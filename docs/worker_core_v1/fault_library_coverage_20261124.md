# 故障库41行覆盖台账（唯一当前口径）

2026-11-24修订。**库行号与公共故障编号是两个命名空间。** 库8是CP padding，不是公共8；库17是转置卷积padding，公共17仅提供相关空间误差现象。库27映射公共15，库41映射公共20，库39映射公共39mr。旧的12行映射已被本表替代，冻结发行物内历史文档不改写。

A公共交付10行；B黄区已有原语17行；C现象复现8行（新纳入30）；D仅V4 1行；E配置2行；F不适用3行。合计41行。黄区能力与状态仅引用回传，不把蓝区审查或示例当作黄区产品通过。

登记按“能力＋命名变体＋部署组合＋模式”一次记录，覆盖库行仅作备注。不得由同一行号自动推导公共机制。下列A类为已交付能力，不意味着本轮最终包已登记。

进度快照：2026-11-24：逐项A2独立复核更新（黄区正式登记未完成）。A–F 是归属，progress_state 是进度，两者独立。以下数字从逐行 JSON 汇总；52、56 行不在统计范围内。

|进度状态|行数|库行|
|---|---|---|
|CLOSED_NO_INJECTION|5|1、4、5、29、36|
|AWAIT_FORMAL_TEST|14|6、15、17、18、20、22、23、26、27、28、30、35、39、41|
|YELLOW_ADAPTER_READY_UNTESTED|8|2、11、14、21、24、31、37、40|
|REFERENCE_ADAPTER_PENDING|4|12、19、32、34|
|HCCL_BITWISE_PENDING|2|25、38|
|FAULT33_EVIDENCE_PENDING|1|33|
|V4_D_PACKAGE_PENDING|1|13|
|ADAPTER_READY_UNTESTED|6|3、7、8、9、10、16|

历史公共证据不等于本轮通过。参考适配器通过 A2 后记 ADAPTER_READY_UNTESTED，标注等待黄区审查；正式测试审查并登记后才记 REGISTERED。第21行按区域寿命选择v1或v2，待黄区确认。

|库行|类|进度|对应能力/变体|机理与范围|部署|证据或待补项|
|---|---|---|---|---|---|---|
|1|E|CLOSED_NO_INJECTION|rope_scaling配置|配置说明，不注入|Qwen3.8 A3 TP2（蓝区A2仅验证）|范围决定，不要求注入|
|2|B|YELLOW_ADAPTER_READY_UNTESTED|黄区weight/faulty_requant、weight/param_perturb（v2）|持久权重/scale；覆盖派生副本与全部读取者|Qwen3.8 A3 TP2（蓝区A2仅验证）|黄区归属表声明已有原语；当前模型适配、产品回执待黄区|
|3|B|ADAPTER_READY_UNTESTED|黄区activation_mismatch/swiglu_divergence；保留配置说明|SwiGLU分支数值差；不是所有模型均有该调用|Qwen3.8 A3 TP2（蓝区A2仅验证）|reference_adapter_a2_20261124.json#row3_gate_fp16_round；回执SHA af269176c362ab2174e6ed3640b9765aa5640536dc2af75e9058dbf083ed689c；原语差异见reference_adapters/README.md|
|4|E|CLOSED_NO_INJECTION|量化校准数据集分布|校准数据分布影响，非运行时故障|Qwen3.8 A3 TP2（蓝区A2仅验证）|范围决定，不要求注入|
|5|F|CLOSED_NO_INJECTION|GPT-NeoX RoPE能力缺失|功能缺失，不是可注入故障|当前A3不适用|范围决定，不要求注入|
|6|C|AWAIT_FORMAL_TEST|公共22 wrong_row|现象：错误cos/sin行；不复现超4096缓存不足|Qwen3.8 A3 TP2（蓝区A2仅验证）|历史公共结果见证据索引；最终产品登记仍须逐组合审查|
|7|C|ADAPTER_READY_UNTESTED|黄区operator_output_value，MLP部分token行置零|现象：贡献丢失；dense模型无MoE超容量|Qwen3.8 A3 TP2（蓝区A2仅验证）|reference_adapter_a2_20261124.json；yellow-reference-hidden-20261124-r1；真实add/rmsnorm消费者，旧metadata-only结果已作废|
|8|C|ADAPTER_READY_UNTESTED|黄区operator_output_value，激活行替换padding/零|现象：有效激活行被零/padding替换；不证明CP restore索引错误|Qwen3.8 A3 TP2（蓝区A2仅验证）|reference_adapter_a2_20261124.json；yellow-reference-hidden-20261124-r1；真实add/rmsnorm消费者，旧metadata-only结果已作废|
|9|B|ADAPTER_READY_UNTESTED|黄区fake_requant saturation|须证明真实scale除法及clip；仅改变反量化乘数不等价|Qwen3.8 A3 TP2（蓝区A2仅验证）|reference_adapter_a2_20261124.json；yellow-reference-quant-20261124-r1独立补充件；原始QMM消费、字节恢复、逐位金丝雀和负例通过|
|10|B|ADAPTER_READY_UNTESTED|黄区fake_requant squeeze；公共Batch已有相关证据|须观测共享/增大scale造成的量化分辨率降低|Qwen3.8 A3 TP2（蓝区A2仅验证）|reference_adapter_a2_20261124.json；yellow-reference-quant-20261124-r1独立补充件；原始QMM消费、字节恢复、逐位金丝雀和负例通过|
|11|B|YELLOW_ADAPTER_READY_UNTESTED|黄区activation/faulty_requant|重量化语义及真实消费者由黄区确认|Qwen3.8 A3 TP2（蓝区A2仅验证）|黄区归属表声明已有原语；当前模型适配、产品回执待黄区|
|12|C|REFERENCE_ADAPTER_PENDING|黄区operator_output_value：compute_logits列错位/命名坍塌|C类现象复现；原机理为lm_head错误量化/量化误差累积，当前lm_head BF16未量化，不宣称原机理成立|Qwen3.8 A3 TP2（蓝区A2仅验证）|已收到xlsx第12行原文；当前A3词表248320，124160是他模历史值；需真实sampler消费、熵/分布变化与恢复，待黄区正式验收|
|13|D|V4_D_PACKAGE_PENDING|黄区router量化噪声变体，经D包外部v1|须观测router概率、top-k与真实路由；加噪不等于专家改变|V4 A3 TP8+EP|D0/D1及V4 router原件待回传；V2-Lite仅预演|
|14|B|YELLOW_ADAPTER_READY_UNTESTED|黄区kv_rope_quant_loss（v2）|KV持久性按请求/页代际恢复|Qwen3.8 A3 TP2（蓝区A2仅验证）|黄区归属表声明已有原语；当前模型适配、产品回执待黄区|
|15|A|AWAIT_FORMAL_TEST|公共15 alias_detach（B档ND）|ND别名断开现象；无私有格式触发条件|Qwen3.8 A3 TP2（蓝区A2仅验证）|最新负责人汇总称A3双owner通过，仍需正式登记回执；15初值风险已完成A2两轮诊断，均未复现历史44次非有限，限制见r741_b1_b6_closure_20261124.md。|
|16|B|ADAPTER_READY_UNTESTED|黄区conv2d_compute_downcast扩展Conv3d|低精度卷积重算需只读真实输入和权重；Conv2d原语不能直接套3D|Qwen3.8 A3 TP2（蓝区A2仅验证）|reference_adapter_a2_20261124.json#row16_conv3d_compute_fp16；yellow-reference-vision-compute-20261125-r1；真实encoder add消费、恢复/金丝雀/负例通过|
|17|C|AWAIT_FORMAL_TEST|公共17 spatial_padding（Conv3d）|仅空间位置/边界误差同类现象；不等价于ConvTranspose输出高度减2|Qwen3.8 A3 TP2（蓝区A2仅验证）|历史公共结果见证据索引；最终产品登记仍须逐组合审查|
|18|A|AWAIT_FORMAL_TEST|公共18 api_half|当前api_half只缩放单元素，非FP16强转原机理|Qwen3.8 A3 TP2（蓝区A2仅验证）|最新附件报告A3双owner通过；最终包登记材料待审|
|19|B|REFERENCE_ADAPTER_PENDING|黄区float_matmul（独立适配，不能并入公共39mr）|B类；累加顺序/精度规则改变需内部计算证据，输出有限微扰只能记现象代理|Qwen3.8 A3 TP2（蓝区A2仅验证）|黄区报告需适配；库19/34/39分别记录累加精度差/融合差/溢出，52/56原描述尚无材料，不能猜测或据此登记|
|20|A|AWAIT_FORMAL_TEST|公共20 buffer_substitute；黄区operator_input_precision_transform|输入内容替换；不声称原kernel地址槽错误。降精度为另一能力|Qwen3.8 A3 TP2（蓝区A2仅验证）|历史公共结果见证据索引；最终产品登记仍须逐组合审查|
|21|B|YELLOW_ADAPTER_READY_UNTESTED|黄区param_perturb（按区域寿命选v1/v2）|若改weight/scale等持久参数必须v2，不能v1局部恢复|Qwen3.8 A3 TP2（蓝区A2仅验证）|黄区归属表声明已有原语；当前模型适配、产品回执待黄区|
|22|A|AWAIT_FORMAL_TEST|公共22 wrong_row|错位置行代理；核对真实rotary消费者|Qwen3.8 A3 TP2（蓝区A2仅验证）|历史公共结果见证据索引；最终产品登记仍须逐组合审查|
|23|C|AWAIT_FORMAL_TEST|公共39 output_proxy|有限值进入消费者前变Inf；不复现baddbmm内部FP16溢出|Qwen3.8 A3 TP2（蓝区A2仅验证）|历史公共结果见证据索引；最终产品登记仍须逐组合审查|
|24|B|YELLOW_ADAPTER_READY_UNTESTED|黄区exponential_perturb（采样site）|修改真实采样噪声，采样除法读取后恢复|Qwen3.8 A3 TP2（蓝区A2仅验证）|黄区归属表声明已有原语；当前模型适配、产品回执待黄区|
|25|B|HCCL_BITWISE_PENDING|黄区独立HCCL工具|独立通信通道；不能伪装成模块v1适配器|Qwen3.8 A3 TP2（蓝区A2仅验证）|黄区归属表声明已有原语；当前模型适配、产品回执待黄区|
|26|A|AWAIT_FORMAL_TEST|公共26 fia_destination|FIA数据写真实residual；不声称kernel真实越界|Qwen3.8 A3 TP2（蓝区A2仅验证）|历史公共结果见证据索引；最终产品登记仍须逐组合审查|
|27|A|AWAIT_FORMAL_TEST|公共15 alias_detach（B档ND）|同公共15能力一次登记，两库行备注|Qwen3.8 A3 TP2（蓝区A2仅验证）|最新负责人汇总称A3双owner通过，仍需正式登记回执；15初值风险已完成A2两轮诊断，均未复现历史44次非有限，限制见r741_b1_b6_closure_20261124.md。|
|28|A|AWAIT_FORMAL_TEST|公共28 denominator_replay|分母重复贡献为错误归约代理，非合法tiling重排|Qwen3.8 A3 TP2（蓝区A2仅验证）|历史公共结果见证据索引；最终产品登记仍须逐组合审查|
|29|F|CLOSED_NO_INJECTION|310P特有硬件条件|当前A3不具备310P条件|当前A3不适用|范围决定，不要求注入|
|30|C|AWAIT_FORMAL_TEST|公共18方差误差命名变体（新增）|固定增益1/sqrt(k)的归一化输出代理；非INTraining训练方差实现|Qwen3.8 A3 TP2（蓝区A2仅验证）|18.variance_over4（k=4、gain=0.5）和18.variance_under4（k=0.25、gain=2）均通过A2/0.26/CANN9.1/Qwen3.8 W8A8/TP2两个owner；见r741_b1_b6_closure_20261124.md。A3待验收。|
|31|B|YELLOW_ADAPTER_READY_UNTESTED|黄区fake/faulty_requant|与黄区重量化能力合并，按命名变体登记|Qwen3.8 A3 TP2（蓝区A2仅验证）|黄区归属表声明已有原语；当前模型适配、产品回执待黄区|
|32|B|REFERENCE_ADAPTER_PENDING|黄区operator_input_precision/compute_downcast扩展Conv3d|输入cast与内部低精度重算须分别记变体；3D/图片待验收|Qwen3.8 A3 TP2（蓝区A2仅验证）|reference_adapter_a2_20261124.json#row32_conv3d_input_fp16：每rank applied/consumed/restored=6，changed=0、changed_elements=0；不额外加扰动|
|33|A|FAULT33_EVIDENCE_PENDING|公共33 paged_decode|A3实际launch/三形状控制未闭合，关闭|Qwen3.8 A3 TP2（蓝区A2仅验证）|r7.5候选解码工具已提供；A3三形状实际回传待补|
|34|C|REFERENCE_ADAPTER_PENDING|黄区float_matmul现象变体（独立登记）|C类；eager无原图编译融合，比较MM与加法路径的数值现象，不用公共39mr结果替代黄区适配器验收|Qwen3.8 A3 TP2（蓝区A2仅验证）|黄区float_matmul需适配；同一次测试仅在证据逐条支持时备注覆盖多行|
|35|B|AWAIT_FORMAL_TEST|合并公共28 denominator_replay；黄区finite_perturb历史保留|数学不等价：分母重放不是仅累加序改变，合并只覆盖错误输出现象|Qwen3.8 A3 TP2（蓝区A2仅验证）|黄区归属表声明已有原语；当前模型适配、产品回执待黄区|
|36|F|CLOSED_NO_INJECTION|BIOS硬件配置|不修改BIOS，不作为本工具注入|当前A3不适用|范围决定，不要求注入|
|37|B|YELLOW_ADAPTER_READY_UNTESTED|黄区rounding_perturb / faulty_requant_rounding；保留配置说明|固定输入与scale，仅改变局部舍入，不全局替换torch.round|Qwen3.8 A3 TP2（蓝区A2仅验证）|黄区归属表声明已有原语；当前模型适配、产品回执待黄区|
|38|B|HCCL_BITWISE_PENDING|黄区独立HCCL工具|独立HCCL通道；引用黄区结论|Qwen3.8 A3 TP2（蓝区A2仅验证）|黄区归属表声明已有原语；当前模型适配、产品回执待黄区|
|39|A|AWAIT_FORMAL_TEST|公共39 matmul_range|需真实非有限结果才覆盖溢出；有限误差只能说明精度损失|Qwen3.8 A3 TP2（蓝区A2仅验证）|历史公共结果见证据索引；最终产品登记仍须逐组合审查|
|40|B|YELLOW_ADAPTER_READY_UNTESTED|黄区kv_update_misdirect（v2）|污染在请求后续token生效，释放/重建边界恢复|Qwen3.8 A3 TP2（蓝区A2仅验证）|黄区归属表声明已有原语；当前模型适配、产品回执待黄区|
|41|A|AWAIT_FORMAL_TEST|公共20 buffer_substitute|同公共20输入覆盖；一次登记两库行备注|Qwen3.8 A3 TP2（蓝区A2仅验证）|历史公共结果见证据索引；最终产品登记仍须逐组合审查|

## 等价性结论与限制

- 库6：公共22给真实rotary计算错误位置行，能够出现位置编码错误。当前上下文1024不能触发原始超过4096的缓存问题；只记现象复现。
- 库17：公共17改变Conv3d padding，能够影响空间边界/位置；ConvTranspose2d的输出高度减2及view报错不由此证明。严格同一现象不成立，只能接受更宽的空间误差代理。
- 库23：39op把有限输出改为Inf并由真实消费者读取，覆盖非有限传播；不证明FP16 baddbmm内部溢出。
- 库30：RMSNorm使用均方根而非中心化方差。乘1/sqrt(k)只在忽略epsilon（或epsilon同倍缩放）时等价于分母统计量乘k；固定epsilon时准确比例是sqrt((s+epsilon)/(k*s+epsilon))。本变体标注输出增益代理。已有18只改一个元素，不能把旧验收当成整体方差误差验收；新变体必须单独测。
- 库15/27：同公共15 B档ND写入丢失，不能恢复“私有格式base”的历史条件。库20/41复用输入内容覆盖；库26为真实目的缓冲区写入；均按已声明边界登记。
- 库28/35：错误分母重放与合法切分/累加顺序变化数学不等价。库35仅记相关错误输出代理；不新建机制补造原理等价。
- 库39：39mr的有限精度变化不必出现Inf；按实际结果选择“溢出”或“有限精度损失”，不由名称推定。

## 配置说明与来源

库1：模型config中的rope_scaling；默认遵循实际加载config。库4：量化校准数据集及分布；默认和实际制品资料未提供，不补值。库6：原报告的4096边界不是当前统一模型默认值，检查实际缓存实现与上下文。

库3：SwiGLU分支变体参数由黄区原语定义；默认值未提供。库19：CLOSE_MATMUL_K_SHIFT，黄区报告A3未设置；未设置不等于值0或1，须核对实际SDK消费者。库37：黄区faulty_requant_rounding的rounding_mode默认为nearest_even，候选trunc/floor/ceil；ASCEND_ROUNDING_MODE在黄区报告中未设置。局部变体不等于改变全局环境变量。

只读核对：在真实Worker进程读取config并计算SHA，打印环境变量是否存在及其原值；黄区本地核对变体签名/默认值。不能从另一容器、历史脚本或默认值补齐缺项。

证据来源：黄区develop原语41行归属文本、黄区原始描述文本及本轮两个任务附件；只记录下列文件SHA，不提交原始接收文件。历史公共回执见yellow_r5_extract_review_20261123.json；最新A3摘要不替代最终包产品登记。原先12行表已撤销。

- `3109d8f2138588ba402607d61b5068efdd7a0a4814100f1d583dde659418272e` — develop原语核查_41条归属与未完成项复核_原文.txt
- `354bdfe7b8014d97c77d36ff9adec577c7da854f66f987c8404411b858730acb` — 故障库原始描述_新增设计与A3复核清单_原文.txt
- `8528a76700c010db5a1c8202c363d6b921feee65357d48d151f530de6748b4f4` — 已粘贴的文本.txt
- `02cbb487ab11226ad96266b2fb45ac43175b2baa8abc36ddd486bf8e5b130665` — 已粘贴的文本.txt

## 与黄区进度冲突（K4）

|行|更正后的口径|尚需验证|
|---|---|---|
|3|activation_mismatch / swiglu_divergence已被列为注入能力，同时保留配置说明；不是“配置类故不适用”|真实SwiGLU分支和MLP消费者|
|9、10|已有fake_requant saturation/squeeze；r7.4.1提供真实DynamicQuant dispatch输入示例|输入变换能否改变实际Q/scale并产生clip/分辨率变化；不能只改反量化乘数|
|16、32|Qwen视觉塔有Conv3d；文本请求没有执行它不等于模型没有Conv|真实图片；Conv2d原语需适配3D；输入cast和低精度重算分别验收|
|12|已收到原文；当前词表248320、lm_head BF16未量化，原错误量化机理不适用，但按C类做输出列错位/坍塌|真实compute_logits→sampler；不篡改持久权重，不声称重现错误量化|
|19|float_matmul独立适配，不能并入39mr|19累加序精度差；34融合前后精度差仅现象；39实际溢出需NaN/Inf证据；52、56缺原文，不推断|

第12行原文身份记录在JSON的row12_source，原件留仓库外。不得继续使用“待提供第12行原文”的历史说明。所有A3正式登记以r7.4.1统一测试为准，旧过程汇总不免测。适配配方见yellow_adapter_recipes_20261124.md。
