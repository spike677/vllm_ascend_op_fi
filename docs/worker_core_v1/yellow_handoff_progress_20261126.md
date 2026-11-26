# 41行交接进度快照（2026-11-26）

来源为fault_library_coverage_20261124.json；A–F归属不变，progress_state是独立维度。本快照不覆盖原台账历史，不把历史公共结果或页面通过改写为REGISTERED。逐行状态和统计由同名JSON生成，当前登记0行。

类别计数：{"A": 10, "B": 17, "C": 8, "D": 1, "E": 2, "F": 3}。
进度计数：{"CLOSED_NO_INJECTION": 5, "FAULT33_EVIDENCE_PENDING": 1, "HCCL_BITWISE_PENDING": 2, "R8_FORMAL_EVIDENCE_INCOMPLETE": 13, "R8_VARIANT_PRODUCT_INCOMPLETE": 1, "V4_D_PACKAGE_PENDING": 1, "YELLOW_LOCAL_ACCEPTANCE_PENDING": 18}，合计41。

|库行|类别|能力/映射|当前进度|
|---:|---|---|---|
|1|E|rope_scaling配置|CLOSED_NO_INJECTION|
|2|B|黄区weight/faulty_requant、weight/param_perturb（v2）|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|3|B|黄区activation_mismatch/swiglu_divergence；保留配置说明|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|4|E|量化校准数据集分布|CLOSED_NO_INJECTION|
|5|F|GPT-NeoX RoPE能力缺失|CLOSED_NO_INJECTION|
|6|C|公共22 wrong_row|R8_FORMAL_EVIDENCE_INCOMPLETE|
|7|C|黄区operator_output_value，MLP部分token行置零|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|8|C|黄区operator_output_value，激活行替换padding/零|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|9|B|黄区fake_requant saturation|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|10|B|黄区fake_requant squeeze；公共Batch已有相关证据|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|11|B|黄区activation/faulty_requant|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|12|C|候选logits错列/换列，原始描述待补|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|13|D|黄区router量化噪声变体，经D包外部v1|V4_D_PACKAGE_PENDING|
|14|B|黄区kv_rope_quant_loss（v2）|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|15|A|公共15 alias_detach（B档ND）|R8_FORMAL_EVIDENCE_INCOMPLETE|
|16|B|黄区conv2d_compute_downcast扩展Conv3d|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|17|C|公共17 spatial_padding（Conv3d）|R8_FORMAL_EVIDENCE_INCOMPLETE|
|18|A|公共18 api_half|R8_FORMAL_EVIDENCE_INCOMPLETE|
|19|B|黄区operator_output_value_transform finite_perturb / float_matmul|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|20|A|公共20 buffer_substitute；黄区operator_input_precision_transform|R8_FORMAL_EVIDENCE_INCOMPLETE|
|21|B|黄区param_perturb（按区域寿命选v1/v2）|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|22|A|公共22 wrong_row|R8_FORMAL_EVIDENCE_INCOMPLETE|
|23|C|公共39 output_proxy|R8_FORMAL_EVIDENCE_INCOMPLETE|
|24|B|黄区exponential_perturb（采样site）|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|25|B|黄区独立HCCL工具|HCCL_BITWISE_PENDING|
|26|A|公共26 fia_destination|R8_FORMAL_EVIDENCE_INCOMPLETE|
|27|A|公共15 alias_detach（B档ND）|R8_FORMAL_EVIDENCE_INCOMPLETE|
|28|A|公共28 denominator_replay|R8_FORMAL_EVIDENCE_INCOMPLETE|
|29|F|310P特有硬件条件|CLOSED_NO_INJECTION|
|30|C|公共18方差误差命名变体（新增）|R8_VARIANT_PRODUCT_INCOMPLETE|
|31|B|黄区fake/faulty_requant|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|32|B|黄区operator_input_precision/compute_downcast扩展Conv3d|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|33|A|公共33 paged_decode|FAULT33_EVIDENCE_PENDING|
|34|C|黄区finite_perturb或公共39 matmul_range|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|35|B|合并公共28 denominator_replay；黄区finite_perturb历史保留|R8_FORMAL_EVIDENCE_INCOMPLETE|
|36|F|BIOS硬件配置|CLOSED_NO_INJECTION|
|37|B|黄区rounding_perturb / faulty_requant_rounding；保留配置说明|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|38|B|黄区独立HCCL工具|HCCL_BITWISE_PENDING|
|39|A|公共39 matmul_range|R8_FORMAL_EVIDENCE_INCOMPLETE|
|40|B|黄区kv_update_misdirect（v2）|YELLOW_LOCAL_ACCEPTANCE_PENDING|
|41|A|公共20 buffer_substitute|R8_FORMAL_EVIDENCE_INCOMPLETE|

库行号不是公共故障编号：库8为CP padding，库27映射公共15，库41映射公共20。15/27、20/41、28/35按能力+变体复用证据，不重复登记。库35的错误分母重放只作相关错误输出代理，不能写成合法归约等价。

YELLOW_LOCAL_ACCEPTANCE_PENDING只表示由黄区本地继续适配/审查/正式测试；不替黄区声明其适配器已通过。库21的v1/v2由区域寿命决定。库12原文已收到：D:/CodexArchives/20261124/yellow-fault-library-row12/库行12_token词表映射坍塌_原文.txt（SHA 12fc560ff7148c964a4fb514e086bfabb19ddd73dd05b88ae7b1a06d0c0347eb）；附件中“原始描述待补”是历史状态，不能继续沿用。现象代理不声称重现原始量化机制。
