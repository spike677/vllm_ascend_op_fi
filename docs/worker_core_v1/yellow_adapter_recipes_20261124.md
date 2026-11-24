# 黄区B/C类适配配方（K4，非适配器实现）

对象为A3 Qwen3.8 TP2、r7.4.1；下表不是已通过清单。蓝区只给出挂点与证据要求，不编写黄区原语/变体。v1注册允许stage=model/sampler/vision，region_kind仅transient；v2使用PersistentSite，region_kind=persistent_weight/persistent_kv，没有stage构造参数。内置能力不使用外部注册。SHA审查、逐rank注册、许可和CLI显式映射均不可省略。

|库行|挂点|stage|region_kind|变换|请求|消费者与恢复|
|---|---|---|---|---|---|---|
|2|v2持久权重|model（语义；PersistentSite无stage参数）|persistent_weight|weight/scale错误量化，派生副本同审|文本|实际QMM读取全部相关权重/scale；独占排空后恢复全部派生对象|
|3|模块输出/真实SwiGLU操作数|model|transient|swiglu_divergence|文本|真实MLP消费分支输出；同stream消费后区域恢复|
|6|公共22 rope|内置model|内置瞬态|wrong_row|文本|实际rotary消费者与恢复；非原超长缓存机理|
|7|MLP模块输出|model|transient|选中token行置零|文本|对应请求行下游读取；只复现贡献丢失，不声称MoE超容量|
|8|激活模块输出|model|transient|选中有效行用零/padding替换|文本|下游消费错误行；非公共8，非CP原restore索引|
|9|DynamicQuant dispatch输入|model|transient|fake_requant saturation|文本|记录实际Q/scale与饱和元素；原QMM读取；输入或临时副本消费后恢复|
|10|DynamicQuant dispatch输入|model|transient|fake_requant squeeze|文本|实际量化级距/小值损失；原QMM读取；不是仅输出乘常数|
|11|量化模块输入/dispatch操作数|model|transient|activation faulty/fake_requant|文本|Q/scale精度变化和QMM读取；恢复输入/副本|
|12|compute_logits返回到真实sampler|model（跨execute/logits生命周期需核对）|transient|logits列错位/命名坍塌|文本|真实sampler读被改列；token/熵/分布变化分开记；不得只挂lm_head并假定后续无变换|
|14|v2 KV写入前/请求页|model（语义）|persistent_kv|kv_rope_quant_loss|文本含decode|真实FIA读取污染K/V；页代际/请求归属；释放前安全恢复，不覆盖新token合法写入|
|16|视觉Conv3d模块输出＋只读输入/权重|vision|transient|按低精度重算Conv3d|图片|输入/weight只读，输出由encoder消费；输出区域和hook恢复|
|17|公共17 Conv3d|内置vision|内置瞬态|spatial_padding|图片|encoder消费者；只覆盖空间边界误差，非转置卷积原机理|
|19|linear模块输出＋真实输入/权重，或内部matmul操作数|model|transient|独立float_matmul累加规则|文本|内部重算证据→真实下游；只改输出数值不可宣称累加序机理|
|21|模块输入/输出；持久参数则v2|model|transient / persistent_weight|param_perturb按对象寿命定类|文本|临时区同stream恢复；weight/scale独占排空并恢复派生副本|
|23|公共39op linear输出|内置model|内置瞬态|有限值变Inf|文本|真实下游消费Inf和恢复；非FP16内部溢出机理|
|24|采样噪声/采样除法|sampler|transient|exponential_perturb|temperature>0、固定seed|原sampler除法实际读取噪声；读完恢复；greedy不制造随机调用|
|25|黄区独立HCCL|不适用外部stage|不适用外部region_kind|黄区已有通信变体|文本TP2|通信调用/跨rank结果和逐位金丝雀；不伪装provider事件|
|30|公共18 norm输出|内置model|内置瞬态|variance_over4 / variance_under4|文本|RMS均方增益代理；整体输出消费者，非精确中心方差|
|31|DynamicQuant/QMM真实操作数|model|transient|fake/faulty_requant反量化位置变体|文本|实际QMM输入/scale及消费；输入位点不支持的返回明确待适配|
|32|视觉Conv3d真实输入参数|vision|transient|输入cast后按已审核dtype/布局消费|图片|原Conv3d读取转换输入；输入参数作用域结束恢复；输出/权重只读示例不等于输入cast已验收|
|34|独立float_matmul重算输出|model|transient|融合前后数值现象|文本|与黄区19适配器相关但逐行出证；不自动沿用公共39mr结果|
|35|公共28 attention|内置model|内置瞬态|denominator_replay错误分母|文本|真实消费者；仅错误输出现象，不证明合法累加序等价|
|37|量化dispatch操作数/局部舍入调用|model|transient|nearest_even/trunc/floor/ceil命名变体|文本|固定输入和scale，实际Q/scale与QMM消费；不全局patch torch.round|
|38|黄区独立HCCL|不适用外部stage|不适用外部region_kind|黄区已有通信变体|文本TP2|同25；逐位金丝雀，独立路径回执|
|40|v2 KV真实写入页/slot|model（语义）|persistent_kv|kv_update_misdirect|文本含decode|错槽影响当前请求后续读取；页释放前恢复，不碰复用页|

DynamicQuant输入示例只证明该位置可接入，不能把任意输入变换都称为“scale饱和/挤压”；黄区必须核验自己的变体与实际量化返回值。Conv3d输出＋只读权重例不自动授予Conv3d输入cast资格。compute_logits不是普通nn.Module.forward，需在现有factory里适配真实调用及sampler读取完成，不得依靠未发生的module hook。挂点覆盖不到或恢复边界无法证明时明确拒绝，不能造调用。

统一计数：observed/eligible/attempted/applied/changed/consumed/restored/failed；changed为调用数，changed_elements独立。v2写入和reader_observations分开。失败或eligible缺覆盖不计正常跳过；OFF三项为区域/资源/新请求金丝雀。当前A3待统一正式测试，无一行因本表获登记。

库19覆盖映射只接受已有描述：19累加顺序精度差；34融合前后数值差（eager现象代理）；39累加器非有限溢出须实测。黄区另提及52、56，但不在本41行表且没有原文，暂记“映射待补”，不自造含义。
