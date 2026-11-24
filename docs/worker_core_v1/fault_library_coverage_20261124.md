# 故障库12行：已有能力覆盖审查

2026-11-24：撤回G1/G2/G3，不新增机制，不替黄区实现变体。按能力＋变体＋组合＋模式登记一次，多行仅作覆盖备注。历史A3 Qwen3.8 TP2验收不自动成为r7.3.1产品登记。

|行|已有能力和变体|机理/现象边界|部署、证据状态|
|---|---|---|---|
|9|黄区activation/param_perturb，缩小pertoken_scale|只有scale进入量化除法且发生clip才能证明饱和；仅改QMM反量化乘数不等价|Qwen，等黄区v1测试|
|10|Batch shared_scale，全批absmax/127|共享极值扩大较小行的量化步长，机制对应；离群数量/位置本身不扩大极值，须实际观察scale和Q变化|Qwen，Batch历史通过，产品待审|
|13|黄区operator_output_value/value_inject，router量化噪声|须观察top-k专家改变；加噪为效果代理，不证明真实量化缺陷|V4，是否纳入D包待负责人确认，黄区需确认模块适配|
|19|黄区float_matmul，参考39mr|累加范围/精度错误可制造数值差，不能自动归因于仅改变累加顺序|Qwen，黄区待测，39mr为相关现象证据|
|23|39op，有限BF16输出写+Inf|覆盖Inf进入消费者并传播，不证明FP16内部溢出。FP16降精度补充项须实际形成Inf|Qwen，39op历史通过，产品待审|
|34|黄区float_matmul/公共39mr|eager错误累加不复现图融合选择，只覆盖精度差现象|Qwen，39mr历史通过，黄区变体待测|
|35|28 denominator_replay|**不等价于合法tiling归约重排**：分母重复贡献改变数学计算，仅覆盖Attention错误输出及传播|Qwen，28历史通过；此限制报负责人，不新增实现|
|39|39op/39mr|op为Inf输出代理；mr须实际出现非有限结果才覆盖溢出现象，有限误差不能写成NaN/Inf复现|Qwen，两能力历史通过，具体非有限现象仍按回执|
|7|黄区operator_output_value，部分MLP token行置零|贡献丢失现象代理，不证明MoE容量或路由溢出|Qwen限定范围待负责人确认，变体待黄区测试|
|8|公共8；Qwen补充22|8须证明错位读到padding；22仅覆盖错误行读取，rotary行不等于token激活，不证明有效token被padding替换|V4原路径待D包；Qwen22历史通过仅作同类现象|
|17|公共17 spatial_padding|padding规则改变，原理对应；限已验收视觉分支与形状|Qwen视觉，17历史通过，产品待审|
|37|黄区activation/faulty_requant，舍入变体|固定输入/scale只改舍入并核对消费者；不保证所有输入出现差异|Qwen，D14升级为注入测试待负责人确认|

## 证据索引

来源：[r5逐份审查JSON](yellow_r5_extract_review_20261123.json)的TP2原始回执SHA256。原件留黄区，是文本提取证明；不是蓝区原件审查。已通过结果能覆盖的部分不重跑；回执未记录的饱和、padding内容、router top-k、非有限值不可补推。

|能力|原始回执SHA256|
|---|---|
|Batch|619e775065094ee30248027e05fef31fd5c84780f095dc93a891cc09272d84d9|
|28|045ea5ec7ca654b052e82ddfaa31a0c1427383a13b4c2ca2bbcd242c4e155cbd|
|22|3ef87e0c8d3747753f7804fcae47ba0554c82ace9429a952470715b3db60885d|
|39op|be859abdfe53b04bd92a13552e624e1375e75ad35a928be06d8b3752b587c15e|
|39mr|d38c5acd78bd2d010f4be475cf89b7e7d66d1f0c52d33cc0204de92fa6118331|
|17|87abc8156953dde89a1bebbac4acd4f960c72ee2cd51adc655685dd3523ca251|

## 第19、37行配置说明

`CLOSE_MATMUL_K_SHIFT`：A3实际值按负责人文本记为“未设置”，非蓝区读取。未设置表示不覆盖SDK默认策略；当前未取得9.1.0消费者源码/默认值证据，不能推定默认数值或具体分支。[Ascend官方固定提交示例](https://gitee.com/ascend/MindSpeed-LLM/blob/3ea3ff7310d95373a1d55ac763cf3e115fc98dda/examples/mcore/llama2/evaluate_llama2_7b_bbh_ptd.sh)设置过1，仅证明示例用法。

第37行 `rounding_mode` 是本文建议的部署字段名，尚非确认的黄区接口/环境变量。默认沿用现有实现，未提供的默认算法不得补成nearest_even；A3未设置覆盖按负责人文本记录。候选nearest_even为最近值且平局取偶；trunc向零；floor向负无穷；ceil向正无穷。测试包括负数及半整数。

核对方法：实际Worker只读打印 `os.environ.get('CLOSE_MATMUL_K_SHIFT')`（null保留）与加载库身份；舍入核对真实部署配置、源码SHA及实际调用round函数。宿主shell不能代表Worker值；不另要求A3只读窗口。黄区尚需确认MLP/router及部分token行适配，蓝区不替写实现。

## 同步决定

r7.4=v2＋33采集器已批准，33适配转r7.5。33使用五条标准：公开候选源码身份、字段边界、至少3形状、每组控制逐位≥99%并定位不一致、任一失败即停。一致率不单独证明解码。第7行限定Qwen、第13行进D包、第37行D14变更仍为本次答复中待负责人决定项。
