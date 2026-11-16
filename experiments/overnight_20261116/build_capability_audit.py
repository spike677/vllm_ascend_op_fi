"""Render separately scoped capability cards from inspected source and evidence.

This is an audit renderer, not a runtime registry or an admission authority.
Historical evidence never promotes the new Worker entry point automatically.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASELINE = 'bfdd6abc1adb5d08b034db7a96ec97ffbec0b9bf'
COMMON_EVIDENCE = 'experiments/provider_adaptation_20261116/validation_summary.json'


def card(name, purpose, implementation, sources, tests, history, roles, operator,
         environment, modification, lossless, effect, recovery, gaps, target, next_action,
         historical_scope='None accepted', historical_levels=(), coverage=None):
    return dict(schema='demoncat.audit.capability.v1', mechanism=name,
        audited_baseline=BASELINE, purpose=purpose, implementation=implementation,
        source_paths=sources.split(), test_paths=tests.split(), historical_paths=history.split(),
        model_adaptation=dict(roles=roles, families=['Qwen', 'GLM', 'DeepSeek'],
            selection='Actual module/call structure, never user-supplied layer names',
            boundary='Family recognition alone neither admits a native call nor certifies this mechanism'),
        operator=operator, environment=environment, native_modification_plan=modification,
        lossless_validation=lossless, effect_validation=effect, off_recovery=recovery,
        current_gaps=gaps.split('|'), sprint_target=target, next_action=next_action,
        historical_validation=dict(scope=historical_scope, levels=list(historical_levels),
            inherited_by_new_worker=False),
        sprint_validation=dict(level='AUDITED', execution_receipt=None,
            raw_statistics_published=False, note='Source/gate review; replace only with explicit new execution receipts'),
        coverage=coverage or dict(single_target='PENDING_CURRENT_WORKER', multi_target='UNVERIFIED',
            every_match='UNVERIFIED', rank_isolation='UNVERIFIED', resource_budget='REQUIRED',
            group_recovery='UNVERIFIED'))


def cards():
    c=[]
    c.append(card('8.restore_offset',
        'MoE还原将目标token的整组expert contribution读成下一token组；indices与top-k权重成对移动。不是index逐元素加1，也不是物理OOB。',
        'LEGACY_SOURCE_RECOVERED_NOT_INTEGRATED', '', 'tests/test_capabilities.py tests/test_lifecycle.py',
        'artifacts/fault8/systemic_restore_offset_20260916/INJECTION_LOGIC.md',
        'MoE router → expert dispatch → token_combine → token_unpermute；需真实请求及token组归属。',
        'torch_npu.npu_moe_token_unpermute(sorted_indices, probs) input boundary',
        '历史Qwen3-VL MoE TP2、INT32 indices/BF16 hidden/probs、T>1完整prefill；新GLM/DeepSeek MoE及A3未验收。',
        '恢复历史SystemicBinding源码并绑定实际combine对象；验证I.reshape(T,K)与P的token-major配对后，source(t)=min(t+1,T-1)，新建两个同形临时device tensors；原生consumer仅调用一次。T=1透传。',
        'Identity offset=0，验证原始映射完整排列、host/device副本一致、原输入未改及native完整输出与B一致。',
        '独立combine分母与API调用数；成对映射hash、实际消费者输入及输出重建。历史模型只比前16列，不能升级全输出审计。',
        '同步后释放临时tensor、owner核对卸载；不声称释放副本撤销故障请求状态。新请求/新缓存OFF对比完整token/logits。',
        'REUSABLE_BUNDLE_INCOMPLETE|REQUEST_MAPPING_UNPROVEN|COMMON_WORKER_PROVIDER_MISSING',
        '已只读取回并核对systemic_extension.py和strategy.py与历史SHA一致；审查配对规则及消费者，保持未接入状态。',
        '补齐fault8_extension/request_scope等依赖和可信scheduler契约；复用已有Worker身份/生命周期而非直接运行旧全局wrapper。',
        '历史Qwen3-VL TP2：48适用调用/rank、B/I/新进程R一致；当前库无可复用worker源码。',
        ('NATIVE_VERIFIED','MODEL_VERIFIED','RECOVERY_VERIFIED')))
    c.append(card('15.alias_detach',
        'as_strided初值正确，但view连接格式转换后的独立存储；真实copy_只更新临时存储，原base仍旧。',
        'IMPLEMENTED_PYTHON_BOUNDARY', 'fault_injection/alias.py',
        'tests/test_alias_auto.py tests/test_cleanup_adapters.py', 'docs/unified_fault_inventory.md',
        '实际module forward中的公开torch.as_strided；动态保留base/view/temporary身份，不指定层名。',
        'torch.as_strided construction → NPU format cast → copy_ → original-base consumer',
        'NPU format29、BF16 base16x16/view8x8/stride16,1/offset17；仅公开Python调用，native内部as_strided未覆盖。',
        '复用NpuFormatProvider将NZ转base format2，必须确认独立storage且初值完全一致；原变量base不得重新绑定。',
        'B原生别名共享；CPU逻辑索引reference检查64个目标元素和全部非目标元素；同步后view初值与copy_结果均正确。',
        'F原base目标元素保持旧值，非目标不变；消费者确实读取同一base。统计首差可在写后base读取/consumer，而非必须as_strided输出。',
        '恢复包装及所有hook；原故障base不能靠disable自动修正，OFF使用同一初始状态重建base并验证。',
        'NEW_ROUTE_NPU_PENDING|NATURAL_MODEL_WRITE_CONSUMER_MISSING|NATIVE_INTERNAL_VIEW_UNOBSERVED',
        '在新入口完成NPU机制级B/F/OFF；自然预训练模型不存在写入链时停止在机制级。',
        '先最小NPU模块证明精确写范围，再寻找真实私有base→view写入→base消费路径；读取权重视图不够。',
        '旧故障27目录实际属15；旧六次NZ机制证据不代表本轮统一入口或自然模型通过。',('NATIVE_VERIFIED',)))
    c.append(card('17.spatial_padding',
        'Conv3d padding参数由0,0,0改为0,7,7；受控空间边界改变，保持已证明的patch输出shape。',
        'IMPLEMENTED_ARGUMENT_PROXY', 'fault_injection/module_faults.py',
        'tests/test_module_faults.py tests/test_cleanup_adapters.py',
        'artifacts/fault17/spatial_padding_hw7_20260920/INJECTION_LOGIC.md',
        '实际标准Conv3d视觉patch embedding；仅类和真实weight/bias/参数关系，不匹配名字proj。',
        'torch.nn.functional.conv3d padding argument',
        'eval标准Conv3d，kernel/stride匹配输入patch，空间16x16、dilation1/groups1、zero padding；NPU新入口待证。',
        '在实际Functional.conv3d唯一调用的padding参数处切换；不修改输入或weight、不人为补写输出。',
        'B透传原参数、原调用一次；核对输入/权重不变和输出N,C,1,1,1；独立CPU卷积参考证明shape。',
        '参数字段可先变化，随后conv3d输出变化；不要求相同参数的kernel计算失败。比较完整输出及下游shape消费。',
        '停止新调用并恢复原Functional.conv3d及hook owner；新同输入forward精确返回B。',
        'NEW_ROUTE_NPU_PENDING|PRETRAINED_VISION_PATH_UNVERIFIED',
        '真实NPU Conv3d机制级参数/数值/shape/OFF验收；真实视觉预训练调用另标。',
        '发现当前视觉模型Conv3d实例并实际触发；文本only模型明确NO_ELIGIBLE_MODULE，不伪造视觉路径。',
        '历史固定视觉patch实验；当前重建入口只有CPU路径验收。',('NATIVE_VERIFIED','MODEL_VERIFIED')))
    c.append(card('18.api_half',
        '标准nn.RMSNorm模块返回值乘0.5的效果代理；不代表CANN Reduce内部累加精度故障。',
        'IMPLEMENTED_RETURN_PROXY', 'fault_injection/module_faults.py',
        'tests/test_module_faults.py tests/test_cleanup_adapters.py',
        'artifacts/fault18/systemic_api_half_20260915/README.md',
        '自动枚举标准torch.nn.RMSNorm对象及实际root forward；自定义QwenRMSNorm/融合Norm不按类名强行接受。',
        'nn.RMSNorm module return', '标准eval nn.RMSNorm，实际浮点返回tensor；自定义/融合类需要专属语义适配。',
        '保留原模块调用及参数；在模块返回边界对真实output实施0.5，按符合条件调用计数。',
        'B增益1且原RMSNorm只调用一次；完整输出、输入/weight和hook顺序确认。',
        'F输出逐元素等于健康输出×0.5；全部符合条件调用计数匹配。msprobe可先看到正常原生Norm输出，下游输入才不同。',
        '移除owned hooks后相同输入再次输出B；不能用enabled flag不检查的旧wrapper。',
        'NEW_ROUTE_NPU_PENDING|CUSTOM_RMSNORM_NOT_SUPPORTED',
        '新入口实际NPU标准RMSNorm模块B/F/OFF、匹配覆盖；自然模型不含标准类型时明确拒绝。',
        '逐个评估自定义Norm实现或实际native API，禁止为扩模型覆盖直接按RMSNorm字符串匹配。',
        '历史Qwen3-VL wrapper；新模块返回代理不继承旧native边界证据。',('NATIVE_VERIFIED','MODEL_VERIFIED')))
    c.append(card('20.buffer_substitute',
        '真实executor Run的语义输入槽临时buffer替换：WeightNz BF16 mantissa与QMM INT8策略分别承诺，原输入存储不变。',
        'IMPLEMENTED_TWO_OPERATOR_PATHS',
        'fault_injection/deployments/buffer20.py fault_injection/deployments/matmul.py fault_injection/native_adapters/legacy20_bridge.inc fault_injection/native_adapters/qmm_auto.inc fault_injection/native_adapters/matmul_auto.h fault_injection/qualification.py',
        'tests/test_worker_extension_providers.py tests/test_native_bindings.py tests/native/test_matmul_auto.cpp',
        COMMON_EVIDENCE+' experiments/production_wiring_20261116/README.md experiments/worker_extension_20261116/validation_summary.json',
        '实际Linear/量化method候选→active module→typed formal/Run slot→唯一稳定cache；QMM与WeightNz各自单独选点。',
        'aclnnMatmulWeightNz(self) / aclnnQuantMatmulWeightNz(x1) at OpExecCache::Run',
        '各自真实v17 ABI/CANN8.5.1构建与库指纹、设备、输入格式和storage边界；不能用WeightNz通过替代QMM。',
        '复用原v17 native算法及typed代理；对每个API独立观察描述符和slot，owner线程仅消费本线程arm，临时输入执行后同步释放。',
        'reference与lossless完整输入storage哈希/完整输出、替换formals、readback、native_calls=1；QualifiedNativeSession保留唯一目标单次门禁。',
        '实际temporary input和真实native输出变化、完整模型/logits、原输入保持；Python可记录未替换的caller输入。',
        'disarm、drain、释放所有临时buffer并核对native idle，再同输入replay/OFF精确对比；不全局reload。',
        'COMMON_WORKER_INJECTION_PENDING|QMM_NEW_BRIDGE_NPU_PENDING|CROSS_THREAD_ARM_RECHECK',
        '共用Worker单点lossless→fault→OFF；WeightNz和QMM分别证明，至少保留Qwen与Llama两结构历史范围。',
        '在当前Worker验证可托管replay、原生scope线程、声明profile和两个API各自receipt；新family只通过实际API复用。',
        '9171f3b WeightNz Qwen3/SmolLM2：591/633调用、3step、OFF；旧QMM真实实验与新qualified桥分开。',
        ('NATIVE_VERIFIED','MODEL_VERIFIED','RECOVERY_VERIFIED')))
    c.append(card('22.wrong_row',
        '原生IndexSelect返回后，将某一目标输出行改为另一合法RoPE source行；是错行输出代理，不是uint32乘法溢出。',
        'LEGACY_EVIDENCE_ONLY', '', 'tests/test_capabilities.py tests/test_lifecycle.py',
        'artifacts/fault22/wrong_row_output_proxy_steps012_20260929/INJECTION_LOGIC.md',
        '实际RoPE实例的cos_sin_cache、positions、aten.index_select调用与npu_apply_rotary_pos_emb消费者。',
        'aten.index_select.default return before actual RoPE consumer',
        '历史BF16 source40960x128、positions0..13、TP2/eager；源10MiB无法触发uint32字节偏移回绕。',
        '先恢复QuickWrongRowExtension并自动绑定cache身份；原生IndexSelect一次执行，设备copy合法source首行到最后输出行，保留原source/index及非目标行。',
        '健康代理不copy或copy同一应读source行，精确比较全部output与source[index]；完整source/index hash保持。',
        '实际错误行被下游rotary消费、Q/K变化及模型输出分叉；不能仅构造错误index或以固定第0层强行验收。',
        'sync/owner检查卸载；新请求新cache重建后token/logits精确恢复。旧错误请求状态不由disable撤销。',
        'REUSABLE_SOURCE_MISSING|CACHE_CONSUMER_BINDING_MISSING',
        '查找历史源码及cache/consumer观察入口；不可得则形成具体自动适配设计与拒绝理由。',
        '查experiments/indexselect_wrong_source_row_proxy/source/quick_extension.py，SHA a9bb9cc575e8df871d0a7056b3ad9ef41a5014052d329cf9e23d128ed5defffb。',
        '旧固定Qwen3-32B/TP2首次prefill每rank一次；input一致/IndexSelect output统计变化，新请求恢复。',
        ('NATIVE_VERIFIED','MODEL_VERIFIED','RECOVERY_VERIFIED')))
    c.append(card('25.reduce_scatter',
        '同输入collective归约次序引起的舍入差异及传播。原设计路线A观察真实后端，路线B重放实际采得的同输入差异；不能换成丢块/重复贡献。',
        'DESIGN_ONLY_CPU_CONTRACT', '', 'tests/test_overnight_evidence_contract.py',
        'docs/unified_fault_inventory.md',
        '实际TP/EP process group、global↔local rank、sequence、reduce op、scatter分片和后续hidden消费。',
        'actual ReduceScatter communication boundary; HCCL/LCCL/NCCL API must first be observed',
        '没有本机制已验收native库；实际group/backend/确定性选项必须观察。CPU三rank模拟不是124双卡通信实测。',
        '先只读观察group/count/dtype/op/stream和collective序号；同输入B多次建立稳定对照。若采到真实差异，可建立与input/group/layout绑定的delta-replay专属代理；无差异样本则拒绝，不随机加噪。',
        '证明每个rank每个贡献恰好一次，receive count×world_size等于send count，成员与调用序号一致，无损代理与原collective完全一致。',
        'B/F统计首差在ReduceScatter输出或未暴露API后的首个consumer；native组/执行与输入同一性另外核验。CPU模拟仅展示加法次序与合法分片。',
        '所有rank一致disarm并drain已发collective；不可单rank跳过collective。等待stream和临时缓冲回收，fresh request恢复；失败隔离整个group。',
        'NATIVE_IMPLEMENTATION_MISSING|ACTUAL_GROUP_UNOBSERVED|SAME_INPUT_DELTA_SAMPLE_MISSING',
        '完成可执行CPU group/rank/分片拒绝合同与数值次序模拟，以及真实native设计；没有实际group不标NPU通过。',
        '从现有worker确认调用的是torch/HCCL还是ATB/LCCL；取得完整调用和group证明再实现对应typed adapter。'))
    c.append(card('26.residual_stomp',
        'QMM真实output-base向相邻live residual存储下移，恢复正常输出与padding后让真实consumer读损坏残差，再恢复残差。',
        'IMPLEMENTED_NATIVE_WRITER',
        'fault_injection/deployments/residual26.py fault_injection/native_adapters/legacy26_qmm_bridge.inc fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_qmm_output_underflow.cpp',
        'tests/test_deployment_providers.py tests/test_native_bindings.py', COMMON_EVIDENCE,
        '同forward真实AddRmsNormBias producer、两个active allocator block、QMM writer与实际residual consumer。',
        'aclnnQuantMatmulWeightNz output-base before native execution',
        'CANN8.5.1/custom AddRmsNormBias；真实相邻active block和padding≤budget；32B W8A8实际是Qwen架构distill。',
        '只接受current-forward存活producer和实际allocator邻接；保留underflow原算法与target/output backing；非邻接透传，不构造假地址。',
        'B不移动输出；无损操作的输出/残差/padding哈希与基线精确一致，当前forward第一层无producer时透传。',
        'native证明实际写入及consumer读取；本轮只读核查rank1 NaN位于__setitem__写前destination，RHS、此前FIA和返回Attention统计有限。未证明分配初始化来源，G9仍阻断，不删NaN。',
        'output、padding、residual逐一精确恢复；pending必须0；消费者未执行不得卸载或继续复用worker。',
        'RANK1_FIRST_DIFF_NAN_BLOCKED|COMMON_WORKER_ADAPTER_PENDING',
        '保留45次/rank真实闭环，独立分析rank1 step0 NaN来源，具备干净对象与原始首差后才提升定位结论。',
        '进一步观察写前destination的实际allocation/初始化生命周期；如需全新采集应保持原始记录并独立复核，不靠删除写前NaN通过。',
        'provider26-final-a2：45eligible writes/rank、pending0、OFF exact；rank0首差下游RoPE，rank1 step0 NaN不可比较。',
        ('NATIVE_VERIFIED','MODEL_VERIFIED','RECOVERY_VERIFIED')))
    c.append(card('26.fia_destination',
        'FIA V3真实输出destination绑定到存活残差；保持正常FIA返回内容，经真实residual consumer传播后恢复。',
        'IMPLEMENTED_NATIVE_WRITER',
        'fault_injection/deployments/residual26.py fault_injection/native_adapters/legacy26_fia_bridge.inc fault20_fast_validation/source/fault26_fia_writer_proxy_v1/fault26_fia_writer_lossless.cpp',
        'tests/test_deployment_providers.py tests/test_native_bindings.py', COMMON_EVIDENCE,
        '实际Attention forward→typed FIA V3 output→同forward residual backing→custom AddRmsNormBias consumer。',
        'aclnnFusedInferAttentionScoreV3 output destination',
        'CANN8.5.1/32B W8A8 TP2，output必须可放入独立live residual storage；不是自然OOB。',
        '复用已验证临时destination/copyback算法，禁止前一prefill残差跨forward变成下一decode的producer；完整consumer receipt后才计completed。',
        'lossless输出替换→正常output精确；残差健康；copyback、sync与consumer生命周期均可证明。',
        '首个prefill差异预计在consumer残差输入；FIA正常返回可一致。decode token已经变时不得以其index_select差异归因新的writer。',
        'consumer后恢复残差并核对copyback；pending0后移除hook，fresh input/cache OFF；未消费隔离worker。',
        'COMMON_WORKER_ADAPTER_PENDING|OTHER_LAYOUTS_UNVERIFIED',
        '复核实际126次/rank完整writer-consumer证据及forward边界修复，避免无意义重复已有模型矩阵。',
        '为共用Extension增加部署factory与trusted replay，保留真实consumer完成/恢复门禁。',
        'provider26-final-a2：126/126completed/rank，首层无producer透传；4/8token/text改变；OFF exact。',
        ('NATIVE_VERIFIED','MODEL_VERIFIED','RECOVERY_VERIFIED')))
    c.append(card('quant.batch_shared_scale',
        '真实DynamicQuantV2从逐token行量化范围变为batch共享范围；这是后端归约范围错误，不是异常prompt自然诱发已证明。',
        'IMPLEMENTED_NATIVE_TLS',
        'fault_injection/quantization.py fault_injection/deployments/batch_scale.py fault_injection/native_adapters/batch_scale_native.cpp',
        'tests/test_quant_auto.py experiments/provider_adaptation_20261116/test_quant_thread_native.py', COMMON_EVIDENCE,
        '实际动态W8A8方法、独立健康decode输入、scheduler request-id→row→token-position/prefix映射，两rank分别证明。',
        'aclnnDynamicQuantV2 native reduction descriptors and scale/Q outputs',
        'CANN9.1、精确vLLM runner SHA、TP2/B8 pure decode、TLS owner；只选首个实际eligible模块。',
        '沿用native量化算术并改变内部[T,H]→[1,T*H]归约范围；不篡改Q/K/V/权重或事后输出。选点和arm按owner线程。',
        '独立健康forward与B/F输入hash、prefix、actual scheduler rows一致；B逐行结果与原算子一致，未选模块透传。',
        '分别核对每rank每request row对应的scale与Q改变；原始statistics按整张量汇总，不用单纯shape[0]代表请求数。',
        '停止新arm、owner线程finish/context释放，重复相同decode OFF exact；异线程finish/disarm必须拒绝。',
        'ALL_ELIGIBLE_MAPPING_UNVERIFIED|PER_REQUEST_SCALE_Q_IMPACT_UNPROVEN|COMMON_WORKER_ADAPTER_PENDING',
        '复用3步TP2模型通过，补每request量化结果证据的摘要/哈希，不从整张量Max推出全部请求都变化。',
        '核对已保存scheduler row映射、scale/Q逐行辅助证据；缺少则限定为全张量统计与请求输入映射已证。',
        'quant-model-a2：每rank3命中；实际scheduler请求/token行映射及selected input一致；3step整批DynamicQuant output0 Max127→98不证明每请求scale/Q均变；6/8logprob变、token不变；OFF exact。',
        ('NATIVE_VERIFIED','MODEL_VERIFIED','RECOVERY_VERIFIED')))
    c.append(card('28.fa.legal_tiling',
        '真实FA S1S2 host选择合法M/nRatio分组，数值变化来自真实浮点计算次序，不保证超过统计阈值。',
        'IMPLEMENTED_STRATEGY_BRIDGE',
        'fault_injection/strategy.py native_execution_strategy/session.py native_execution_strategy/adapters/flash_attention.py',
        'tests/test_strategy_bridge.py', 'docs/unified_fault_inventory.md',
        '实际共享Attention回调与真实npu_fusion_attention_v3参数；请求长度映射独立提供，不从模块名或batch猜。',
        'FlashAttentionScore S1S2 native host policy / actual launch',
        '证书绑定CANN9.1/910B4、B1/B8、Q16/KV8、S2048、D128、FP16及无dropout；不支持TND/复杂mask。',
        '保留StrategySession、NativeFAInvocation和实际observe→choose→host重算；M-only、ratio-only、joint分别归因。不得直接写raw字段骗profile。',
        '原生策略观察与未改变策略launch一致；固定同一完整Q/K/V、逐元素输出与关闭策略恢复。',
        '实际raw/POD/launch与完整输出、msprobe固定容差独立定位。策略切换成功但数值无差异仍保留有效否定结果。',
        '撤销scope并重新运行原策略，对实际launch及完整输出确认恢复。',
        'MODEL_NATIVE_PROVIDER_NOT_WIRED|COMMON_WORKER_ADAPTER_PENDING',
        '核对现有56+4算子矩阵和模型路径，补共用模型forward到策略session的真实绑定可行性。',
        '从实际FA API观察构建NativeFAInvocation，只有证书覆盖形状能执行；不同FIA路径不能复用此通过状态。',
        '历史FA固定shape机制矩阵；当前桥CPU验证不代表新共用Worker实机。',('NATIVE_VERIFIED',)))
    c.append(card('28.fia.legal_tiling',
        'FIA健康合法模板之间切换M/S2分组；不改变累加精度或输出值，不与分母重放错误合并。',
        'IMPLEMENTED_PINNED_ATTENTION_ADAPTER',
        'fault_injection/fia.py fault_injection/attention.py', 'tests/test_fia_masks.py tests/test_fault_contracts.py',
        'docs/unified_fault_inventory.md', '自动共享Attention注册入口，实际Q/K/V、causal mask、scale、cache/prefill语义。',
        'FIA host coupled_base512 → coupled_m128 (healthy templates)',
        '精确vendor SHA，FP16/B1或B8/Hq16/Hkv8/S2048/D128/causal；QK与V维度不同的MLA拒绝。',
        'B M/S2=512，F M/S2=128；正常tiling key109...；原host派生循环/尾块/workspace，legal路径绝不选replay key119...。',
        '完整input与健康base策略nativelaunch相符；同shape原实现执行；OFF原策略精确恢复。',
        '记录真实分组、完整输出和固定容差统计。历史合法策略最佳Norm变化不足0.1%，不可用新分母故障替补通过。',
        '恢复Attention注册及模型config；清除owned临时策略；重新建立同一cache/input再比B。',
        'NEW_ROUTER_NPU_PENDING|FIXED_GEOMETRY_ONLY|STATISTICAL_MAGNITUDE_TARGET_UNMET',
        '保留合法切分数值无超阈值结论，审计通用router是否确实覆盖真实FIA且无语义改写。',
        '用当前router在证书形状验证一次原生launch/恢复；新增形状须新增合法模板及独立证据。',
        '历史固定Qwen3合法切分实验；当前router仅CPU合同。',('NATIVE_VERIFIED',)))
    c.append(card('28.fia.denominator_replay',
        '第二S2块重复首块online-softmax分母贡献的明确计算错误；受控新机制，非历史合法tiling误差。',
        'IMPLEMENTED_PINNED_ATTENTION_ADAPTER', 'fault_injection/fia.py fault_injection/attention.py',
        'tests/test_fia_masks.py tests/test_fault_contracts.py', 'docs/unified_fault_inventory.md',
        '与FIA合法路径共享结构/真实QKV发现，机制ID、kernel key及验收完全独立。',
        'FIA native online-softmax denominator accumulation',
        '固定vendor/FP16/B1或B8/Hq16/Hkv8/S2048/D128/causal；历史Qwen28层，不声称任意GLM/MLA。',
        'B/F均M128/S2=128，B healthy key109...400000，F replay key119...400000；不事后篡改attention/logits。核对重复贡献次数及合法key覆盖。',
        '关闭replay且同一分组，原FIA输出一致；实际template/host/key来源匹配，禁止把合法tiling的不同M当唯一变量。',
        '独立原始statistics/mix按rtol1e-5/atol1e-8找首差，输出Norm等幅度、完整logits辅助传播。0.7968%只属于历史冻结输入。',
        '关闭replay并重建原scope，OFF完整输出精确、临时策略与hook均回收；出错隔离。',
        'NEW_ROUTER_NPU_PENDING|GENERALIZATION_UNVERIFIED',
        '审计真实累计边界/模板、重复次数、恢复与新router门禁；不重复声称旧数值是本轮采得。',
        '取得匹配vendor源码/manifest并在新router单目标实测，再评估覆盖扩展；无matching build拒绝。',
        '历史完整Qwen3/B8/28层/3step首层FIA输出Norm0.7968%变化与OFF；新routerCPU通过不继承。',
        ('NATIVE_VERIFIED','MODEL_VERIFIED','RECOVERY_VERIFIED')))
    c.append(card('33.paged_decode',
        '分页增量Attention合法SInner分组8192→1024改变真实归约路径；不是dense prefill的M分组。',
        'IMPLEMENTED_NATIVE_PAGED_PROVIDER', 'fault_injection/deployments/paged33.py fault_injection/native.py',
        'tests/test_deployment_providers.py tests/test_native_bindings.py', COMMON_EVIDENCE,
        '真实Attention模块、DynamicCache、每请求KV page table及存活页；所有值逆映射证明后执行。',
        'paged IncreFlashAttention native host SInner / actual launch',
        '固定CANN9.1审计bundle/B8/Q1/KV16384/H16/KV8/D128/FP16/page128；teacher decode。',
        '只在真实paged kernel host分组选择边界切换，重算合法派生tiling；page table、KV和输出不人工扰动。',
        'prefix16383后真实1token输入；全值page pack→inverse等于原cache，B/OFF同pagedbackend、原输入保持。',
        '28层×3轮=84次，每次含8请求；独立call_id/pid/tid/time、POD/raw、launch核对。固定容差首差下游mean，不唯一IFA。',
        '停策略并恢复同一pagedbackend原生参数；fresh cache clone OFF exact；退出route恢复原后端但不声称SDPA binary等价。',
        'UNIQUE_FIRST_DIFF_AT_IFA_UNPROVEN|COMMON_WORKER_ADAPTER_PENDING|CONTINUOUS_GENERATION_UNVERIFIED',
        '复核完整page/request/storage/launch证据与统计定位限制，不把84次说成每个请求84次单独launch。',
        '先确认整张量统计为何FIA不可见或低阈值，再扩更多decode长度；持续生成需独立验收。',
        'provider33-model-a2：84/84策略切换，logits maxabs0.02734375、Top1不变、OFF exact；first mean.3 Max .05369125679→.05369321257。',
        ('NATIVE_VERIFIED','MODEL_VERIFIED','RECOVERY_VERIFIED')))
    c.append(card('39.matmul_range',
        '将真实Matmul两个输入替为有限BF16抵消模式，数学结果0但实际计算可出现非有限；不强写输出，也未证明accumulator类型。',
        'IMPLEMENTED_NATIVE_AND_WORKER_BINDING',
        'fault39/native/runtime.cpp fault39/native/metadata_proxy.cpp fault39/src/demoncat_fault39/backend.py src/demoncat/runtime/vllm_ascend/worker_extension/providers/fault39.py',
        'tests/test_worker_extension_providers.py fault39/tests/test_contracts.py',
        COMMON_EVIDENCE+' experiments/worker_extension_20261116/validation_summary.json',
        '实际aclnnMatmul typed self/mat2/out与稳定唯一cache；worker/rank/request/model身份，不以down_proj名称替API。',
        'aclnnMatmul OpExecCache::Run temporary input addresses',
        '独立ABI v1/Config368，BF16 ND/合法stride，精确header/library/profile；8.5.1/9.0/9.1/9.2分别构建。',
        '复用独立Fault39 native与共用Extension路由，单目标/单次/owner代次；self/mat2临时有限±2^e，真实Run后sync并释放。',
        'self-only/mat2-only/dual lossless、原input backing hash、完整输出及实际profile绑定；旧receipt不能授权新构建。',
        '实际输入有限、原Run输出范围异常、原input不变；独立statistics首差后再关联native。无异常保持RANGE_ANOMALY_NOT_OBSERVED。',
        'disable/drain/nativeidle，fresh-request缓存和response/logits恢复；共用Worker old owner在重建后失效，不自动rearm。',
        'COMMON_WORKER_REAL_INJECTION_PENDING|ALL_ELIGIBLE_UNQUALIFIED',
        '在已有profile和同Worker trusted replay完成共用Extension单目标lossless/单次故障/OFF；保持独立ABI。',
        '不能以mock回放或standalone旧TP2证据授予共用Worker闭环；实际调用和完整输出回执是门槛。',
        '独立包四实际CANN算子通过；8.5.1 QwenTP1/TP2、9.1 QwenTP2通过。bfdd共用Worker只证明attach/拒绝/正常恢复。',
        ('NATIVE_VERIFIED','MODEL_VERIFIED','RECOVERY_VERIFIED')))
    c.append(card('39.output_proxy',
        '真实Matmul正常输出后，向已证明的真实output storage写入+Inf/-Inf/qNaN位模式；明确的效果代理，与matmul_range独立。',
        'LEGACY_NATIVE_IMPLEMENTATION_ONLY',
        'fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/buffer_substitute_preload.cpp',
        'tests/test_capabilities.py tests/test_native_bindings.py',
        'fault20_fast_validation/reports/fault39_float_matmul_output_nonfinite_proxy_validation.md',
        '实际Matmul的typed output slot、storage与downstream consumer，不能借range输入替换。',
        'post OpExecCache::Run synchronization / original output storage writeback',
        '历史v19/CANN8.5.1/BF16[1,5120]/rank0/decode一次；独立Fault39 ABI没有这个模式，共用Worker必须拒绝。',
        '新增独立语义provider设计：在真实Run后只选允许的prefix元素，先证全finite/边界，再设备写回并readback；不自动接到现有range模式。',
        '无损写回原output比完整bytes；source/output地址与span由typed证明，原输入保持；未证明output backing则拒绝。',
        '精确验证非有限type/count及非目标字节保持、consumer实际读取；msprobe可先看到native正常值，下游才变。',
        '同步disable并释放owner；output故障已传播不能靠改flag撤销，fresh request/cache恢复；强制失败隔离。',
        'DEDICATED_HOST_CONTRACT_MISSING|NEW_WORKER_PROVIDER_MISSING|CURRENT_STORAGE_PROOF_MISSING',
        '明确独立Provider设计、API拒绝、历史源码归属及验收矩阵；不为凑数量偷用matmul_range。',
        '增加专属Host合同和typed output观察者，再做单次无损writeback→故障→fresh-request OFF验证。',
        '旧Qwen32B v19：prefix1/5/51/512 BF16非有限readback通过；不是当前共用Worker或新四版本通过。',
        ('NATIVE_VERIFIED','MODEL_VERIFIED','RECOVERY_VERIFIED')))
    c.append(card('41.tail_column',
        '真实cache writer完成后改变已证明的live tail列，后续attention读取；受控cache状态代理，不声称自然OOB。',
        'DESIGN_ONLY_DISPATCH_BLOCKED', '', 'tests/test_capabilities.py tests/test_lifecycle.py',
        'fault20_fast_validation/reports/fault41/README.md fault20_fast_validation/reports/fault41/fault41_state_cache_tail_corruption_roadmap.md',
        '实际dispatcher→cache writer→完整storage/slot→请求/KV page→later consumer；地址相近不够。',
        'actual post-write cache tail boundary not yet intercepted',
        '旧.so已加载但真实torch_npu/ATB dispatcher未被拦截；所有新CANN/SoC/provider组合未验。',
        '先只读dispatcher证明；从真实writer元数据导出tail范围并持有allocation。无损保留原尾列；若可支持再单次改值、sync、真实consumer、精确restore。',
        '零修改observer与B一致，完整cache/tail身份和生命周期可验证；输出简单挂钩不能当cache writer命中。',
        'writer后tail发生预期变化且consumer读取同一存储，统计首差与读后传播对应；KV共享/请求混合不明拒绝。',
        '停止新writer、等待consumer/stream、恢复完整tail并核对非目标cache；共享cache不可独立恢复时隔离worker。',
        'DISPATCHER_NOT_INTERCEPTED|CACHE_STORAGE_PROOF_MISSING|CONSUMER_MISSING|NATIVE_PROVIDER_MISSING',
        '检查真实调用/源码，保留blocked事实，输出可执行只读观察设计而非伪造tail故障实现。',
        '从实际wrapper/dispatcher观察合法cache写API，完成只读存储身份闭环后才能开发故障路径。'))
    return c


GATE_FIELDS = {
    'G0': ('实现及来源', 'implementation'), 'G1': ('环境与ABI', 'environment'),
    'G2': ('模型/逻辑层', 'model_adaptation'), 'G3': ('实际算子路径', 'operator'),
    'G4': ('目标身份', 'native_modification_plan'), 'G5': ('无损门禁', 'lossless_validation'),
    'G6': ('实际单次干预', 'native_modification_plan'), 'G7': ('现象观察', 'effect_validation'),
    'G8': ('OFF/恢复', 'off_recovery'), 'G9': ('独立B/F统计', 'independent_statistics'),
}


def baseline_blob(path):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', BASELINE + ':' + path])


def main():
    from sys import path
    path.insert(0, str(ROOT))
    from fault_injection.catalog import list_faults
    rows=cards()
    assert {r['mechanism'] for r in rows} == {r['name'] for r in list_faults()}
    output=ROOT/'profiles/capabilities';output.mkdir(parents=True,exist_ok=True)
    docs=ROOT/'docs/overnight_fault_cards';docs.mkdir(parents=True,exist_ok=True)
    matrix=[]
    for r in rows:
        r['independent_statistics']=dict(task='statistics',level='mix',summary_mode='statistics',
            rtol=1e-5,atol=1e-8,baseline_reference=True,first_diff_before_native_correlation=True,
            preserve_null_nan_inf=True,preserve_actual_steps=True,raw_statistics_publish=False,
            limitation='Statistics are tensor summaries, not per-element equality or a unique physical cause')
        r['source_manifest']=[]
        for p in sorted(set(r['source_paths']+r['test_paths']+r['historical_paths'])):
            if p == 'tests/test_overnight_evidence_contract.py':
                b=(ROOT/p).read_bytes()
                r['source_manifest'].append(dict(path=p,scope='sprint_cpu_design_test',
                    baseline_commit=None,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
            else:
                b=baseline_blob(p)
                r['source_manifest'].append(dict(path=p,scope='baseline_source_or_evidence',
                    baseline_commit=BASELINE,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
        if r['mechanism']=='8.restore_offset':
            r['recovered_source_audit']=dict(read_only=True,executed=False,
                manifest='experiments/overnight_20261116/legacy_source_audit.json',
                native_acceptance_inherited=False)
        if r['mechanism']=='26.residual_stomp':
            r['sprint_diagnostic']=dict(path='experiments/overnight_20261116/fault26_nan_analysis.json',
                kind='READ_ONLY_EXISTING_CAPTURE',native_calls=0,
                result='NAN_IN_PREWRITE_DESTINATION_WITH_FINITE_RHS_AND_ATTENTION',
                independent_first_difference='EARLIEST_DIFFERENCE_UNPROVEN',
                allocation_initialization_proven=False)
        r['gates']={k:dict(name=label,requirement_field=field,
            status='AUDITED' if k=='G0' else 'EXECUTION_EVIDENCE_REQUIRED',
            accepted_for_current_worker=False) for k,(label,field) in GATE_FIELDS.items()}
        # New CPU design tests are not in the baseline. Reference them explicitly
        # without pretending their bytes existed in the historical commit.
        r['audit_scope']='Source and gate review at baseline; execution receipts are a separate overlay'
        (output/(r['mechanism']+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        headings=[('目的与语义','purpose'),('当前实现','implementation'),('模型与逻辑角色','model_adaptation'),
            ('实际算子边界','operator'),('环境/版本/硬件','environment'),('具体实施','native_modification_plan'),
            ('无损门禁','lossless_validation'),('故障效果','effect_validation'),('OFF与恢复','off_recovery'),
            ('本轮目标','sprint_target'),('当前缺口','current_gaps'),('下一步','next_action')]
        lines=['# '+r['mechanism'], '', '**本卡是源码及门禁审计；当前Worker实测结果另行绑定回执，不继承旧实验PASS。**', '']
        for title,key in headings:
            value=r[key]
            if key=='model_adaptation':value=value['roles']+' Qwen/GLM/DeepSeek均须以实际结构及调用证明，不按品牌直接放行。'
            if isinstance(value,list):value='；'.join(value)
            lines.extend(['## '+title,'',str(value),''])
        lines+=['## 历史证据及来源','',r['historical_validation']['scope'],'',
            '历史等级：'+(', '.join(r['historical_validation']['levels']) or '未验收')+'。', '',
            '历史恢复、当前Python/Native入口与共用Worker接线分别验收。本文不自动授权all_eligible/every_match。','',
            '完整G0–G9、覆盖资格、源文件SHA与统计合同见[机器可读能力卡](../../profiles/capabilities/'+r['mechanism']+'.json)。','']
        for p in r['source_paths']+r['test_paths']+r['historical_paths']:
            lines.append('- ['+p+'](../../'+p+')')
        (docs/(r['mechanism']+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        matrix.append(dict(mechanism=r['mechanism'],implementation=r['implementation'],
            audit_level='AUDITED',current_worker_level='UNVERIFIED',
            card='profiles/capabilities/'+r['mechanism']+'.json',
            model=dict(adaptation_targets=['Qwen','GLM','DeepSeek'],
                       verified_scope=r['historical_validation']['scope'],
                       family_membership_grants_support=False),operator=r['operator'],cann=r['environment'],
            soc='Exact observed SoC required; no A2/A3 interpolation',
            gate=r['current_gaps'],cpu='Pending current sprint receipt overlay',
            native=r['historical_validation'],bf='See scoped historical report; no new raw statistics in Git',
            off=r['off_recovery'],coverage=r['coverage'],next_action=r['next_action']))
    report=dict(schema='demoncat.audit.matrix.v1',baseline=BASELINE,mechanism_count=len(rows),
        status='AUDITED_17_SOURCE_AND_GATES',native_executions_in_this_audit=0,
        historical_evidence_inherits_current_acceptance=False,mechanisms=matrix)
    (ROOT/'experiments/overnight_20261116/audit_matrix.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (docs/'README.md').write_text('# 17条机制能力卡\n\n每张卡保留独立语义、真实边界、12项方案、G0–G9与覆盖资格。历史证据与本轮执行回执分开。\n\n'+
        '\n'.join('- ['+r['mechanism']+']('+r['mechanism']+'.md)' for r in rows)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='AUDITED',mechanisms=len(rows),native_calls=0)))


if __name__=='__main__':main()
