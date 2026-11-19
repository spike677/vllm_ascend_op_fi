# 当前产品交付范围与接线状态（2026-11-19）

本轮继承 `0e3df66c5d11d6d66946c63567a79097eabd4757` 及其后继修改，未回退、
覆盖原件或把旧失败改为成功。本文取代旧文档中的后续研究清单；历史 C2/C3 结论仍保留。

**当前未完成实际主项目闭环。** 蓝区只有 Worker Extension 和文档化宿主契约，
没有黄区现役 Catalog、Controller、推理调度调用方与结果界面源码。
公共接口/CPU 测试不等于该主项目已接通，当前仍为 `INNER_FRAMEWORK_INTEGRATION_PENDING`。
也没有把独立 HF 路径、实验 runner 或模拟 Host 标为主项目。TND28 还有故障实现工作，
不是只欠主项目验收。

## 本次实际代码变化

- `fault_injection/actions.py` 抽取实际缩放、独立存储转换、行写入。
  RMSNorm 返回和18模块路径复用缩放；IndexSelect/RoPE 复用存储和行写；
  15把真实格式转换交给同一独立存储检查。公共动作不固定 NZ/几何，15适配器原限制仍保留。
  不创建另一套 owner、回滚或恢复状态机；Native 精度/地址/分母等语义未用 Torch 缩放替代。
- `effect_summary.py` 从已有 CPU readback 汇总完整变化元素数、有限最大差和至多4个样例
  （公共上限8）；Inf/NaN 按类型计数，FP64 差值溢出显式记录。无新的设备读回、同步、
  msprobe 或张量文件。该公共实现已接入真实 RMSNorm 返回路径和 `TensorServingProvider.query()`。
- `current_request_effect` 仅从当前已有请求记录生成，包含 request、phase、invocation、
  输入身份摘要和实际位置；不会比较不同请求，也不把故障记录填进 OFF/健康请求。
  `MATCHED_SAME_RETURN_VALUES_OBSERVED` 只说明这些观察值已绑定；无摘要的命中为
  `EFFECT_NOT_VERIFIED`，截断位置数单独计数。实际消费者是否观察到独立显示。
  `recovery_proven=false` 不由数值摘要升级：健康恢复仍看原 Runtime/provider 的 OFF 和健康请求。
  tensor、普通量化及39复用 `current_completed_row` 处理尚未完成的新请求；pending OFF/health
  不再借用旧F效果。历史 `last_fault_effect`/`fault_effect` 独立保留。这是查询显示修复，
  不改变原注入、Native或恢复，新增窗口CPU契约已通过。
- `delivery_scope.py` 是静态交付说明，并在 catalog、能力查询和 Worker manifest 中统一返回。
  `live_deployment_status=NOT_EVALUATED` 明确区别于实际 Worker admission。
  历史清单仍有17条，公共工厂16条；**工厂数不是支持数**。25在创建 provider 前明确拒绝。
- 当前公共39只接受 `c1 / max_live_leases=1`，33拒绝整组 Native driver，
  继续复用 Source105 单点实现。旧组实现仍供历史研究/测试直接引用，当前公共入口不自动升级。
  下一独立任务使用新 Runtime/provider 作用域；没有重置 `_fault_attempted` 或删掉正常门禁。
- 原三安装单元、`FaultConfig` 三字段、HostAdapter、WorkerFaultClient 与 Runtime 所有权不变。
  源码交付工具要求一起包含公共实现、相关契约测试和本文，防止漏传新依赖。

**本轮进一步拆出的运行时**：39新增 `runtime_only` 单任务入口，通过真实公共 registry
选择 `Output39TaskProvider`，普通作用域不创建目录、不运行完整 census。实际一次请求中
最多16个投影候选的有界观察选出一个真实消费者目标；normal/F/OFF/health由外部真实调度
请求驱动。旧39资格实验仍保留为显式历史入口。
Batch新增普通W8A8投影路径，不等待routed GMM1；现Native ABI仍要求六个真实batch的
normal/reference/lossless/calibration_off/F/OFF，不宣称已压缩为三个batch。
详见[39单任务](output39_single_task_runtime.md)和[普通量化](quant_ordinary_projection.md)。

**仍待拆分的执行代码**：现有 TensorServingProvider、Source105仍包含资格/验收阶段协议，
不能把它们整套包装成普通产品调用。普通请求执行与验收驱动的全面拆分尚未完成。
未改变现有正常资格检查，也未默认新增全目标 census 或采集。

## 当前逐项矩阵

“既有通过”只指链接中的特定 Worker/模型/构建。以下所有“主项目效果/OFF”均未验收，
不得据此声称新模型、新版本或黄区接线已通过。

| 能力 | 本轮状态 | 实现边界／适用部署 | 实际项目效果 | OFF恢复 | 具体阻塞及下一改动 |
|---|---|---|---|---|---|
|8 restore_offset|保留已验收范围|实际 MoE restore/routing/consumer|未接线验收|原范围通过；主项目待验|宿主接线与运行时/验收拆分|
|15 alias_detach|当前部署不适用|真实共享存储→视图写→原base消费|无适用通路|原机制证据保留|不扫描/不造辅助模型；仅有新真实通路才重开|
|17 spatial_padding|保留 Torch C2/C3|真实视觉模型图像请求，单投影Conv3d padding|未接线验收|原范围通过；主项目待验|接入已接受 Torch 边界，不要求新Native/E5|
|18 api_half|共用动作及查询摘要已实现|实际 RMSNorm 返回 gain代理/consumer|CPU契约通过；主项目未验|CPU与原范围恢复；主项目待验|接线、执行/验收拆分和变更后对应实机检查|
|20 buffer_substitute|保留已验收范围|匹配WeightNz/QMM ABI、live slot|未接线验收|原范围通过；主项目待验|宿主按实际部署选点，保留版本界限|
|22 wrong_row|共用行/存储动作已接入|实际 fused RoPE index/consumer|CPU契约通过；主项目未验|原范围保留；主项目待验|接线和变更后对应实机检查|
|25 reduce_scatter|不支持，已停止|无替代通信输出破坏|不适用|不适用|无本轮后续实现|
|26 residual_stomp|选择已有Native单点路线|真实邻接、QMM writer和residual consumer|未接线验收|原单点通过；主项目待验|目标部署邻接不成立则明确不适用，不追135完整集合|
|26 fia_destination|保留通过范围|实际FIA→当前请求残差→consumer|未接线验收|原范围通过；主项目待验|迁移时重新绑定，不继承裸地址|
|Batch shared_scale|普通W8A8新入口单点实测及原件复核通过|Qwen3-0.6B W8A8，TP1/B8首decode，自动qkv投影；六个当前作用域batch|Source118的Q有5565/8192元素变化、scale有7/8变化，实际QMM消费；主项目待验|同Worker OFF的Q/scale及完整hidden/logits哈希精确恢复，前后健康请求一致|实际宿主接线；既有Native ABI仍需作用域校准，不等待routed GMM1|
|28 FA legal|仅历史保留|旧FA限定范围|本轮不做|旧证据保留|退出当前关键路径|
|28 FIA legal|仅历史保留|旧FIA限定范围|本轮不做|旧证据保留|退出当前关键路径|
|28 denominator_replay|TND kernel/host已构建；整条实现未完成|真实TND在线softmax分母重复累计|未验|未验|原物理核调度全部FD已拒；新22项矩阵、SDK及10个实际key构建通过，prepared-input绑定和正常对照待完成，F关闭|
|33 paged_decode|采用Source105单点|真实paged decode，受控S2策略扰动|原单点有内部/logits变化；主项目未验|原单点精确；主项目待验|普通运行/验收拆分与宿主接线；整组路线停止|
|39 matmul_range|保留已验收范围|匹配Native MatMul有限输入/算术边界|未接线验收|原范围通过；主项目待验|宿主接线，不声称全部8.x/9.x|
|39 output_proxy|新独立单任务运行时已接公共registry并实测|finite检查、实际consumer、单次F及恢复|Source116实际1元素变Inf，SwiGLU消费，首token变化；主项目未验|同Worker OFF/health完整hidden/logits哈希恢复一致，资源归零|实际宿主接线；累计C3不再必做|
|41 tail_column|保留已验收范围|实际live KV尾列写读，共用postwrite后端|未接线验收|原范围通过；主项目待验|宿主接线，保留cache语义|

已有成果与准确版本见 [原范围决策](../../experiments/worker_core_v11_20261118/CONVERGENCE_DECISIONS.md)、
[33 Source105](../../experiments/worker_core_v11_20261118/ATTENTION33_SOURCE105_SINGLE_ACCEPTANCE.md)。
39原累计失败、33外部stop部分原件、28 source113零Native输入观察均保留，未升级为成功。

39新入口的[独立原件复核](output39_source116_single_task_review.md)记录：自动观察3个实际投影后
选择首层`gate_up_proj`，单元素`0.96484375 → +Inf`；首token `576/" The" → 0/"!"`，
OFF与health均恢复`576/" The"`。仅验证单请求首token，不授C1/C2/C3/E5或主项目资格。
Source114首个normal的`__class__`闭包误拒仍保留为0F失败；新限定修复没有放宽Native有限值检查。

普通量化[Source118独立原件复核](review_quant_ordinary_source118.md)确认在同一Worker完成
normal/reference/lossless/calibration_off/F/OFF及前后健康请求，
恰好一次F。第8条请求第二token为`374 → 653 → OFF 374`，其余7条的第二token及全部首token不变；
这不是连续长文本或全部模型资格。六组输入摘要一致，实际Native为当前正确域helper后执行共享域，
Q/scale共同发布；最终fault-owned资源及hook/owner释放。固定10,682,368字节的元数据journal
是常驻诊断区，不把“故障资源归零”说成进程总内存归零。
独立复核验证了原件哈希、调用关系及固定源码执行守卫；没有离线重算未下载的完整私有张量。
Source116加载失败和Source117 reference零admission/零F失败均保留。

TND新增[真实输入观察链](tnd28_prepared_input.md)只记录一个匹配目标的原始typed调用，
尚不授权F。Source119的` tnd28-input-observer-build-a1`已通过无设备C++观察合同和原22项
SplitCore检查，但嵌套CMake误拆编译launcher参数，在host包构建时失败，未执行NPU。
失败completion SHA为`fa11d608d03270c42cbf70dbcc6fd104d65b0a56086618fbd7752837f6e4bec6`，
原件保留。后继修复将固定参数放进单一可执行launcher；47项相关CPU通过，须新目录重建。
独审发现的false-LSE占位关联、摘要与原RPC连接及首错保留问题也须修复后才进入正常实测。

## 每项通过及停止条件

矩阵中的下一改动只针对声明范围。通过需实际主项目调用→真实请求效果→对应OFF/健康恢复，
并使用共用代码；不能仅靠注册、arm或命中计数。主项目源码不可达时保持“实现就绪/接线未验收”，
当前28则是“故障实现未完成”。15当前不适用、25不支持、28 legal历史保留均不算支持项。
执行失败保留原件；没有实时匹配信息时不归因于故障。

主项目需要提供已有 Catalog/Controller、normal enqueue 调度及结果出口的真实调用点，
接入现有 HostAdapter 即可继续；不新建模型runner或Worker框架。这个外部缺口此前已确认，
本轮不重复索取不存在的路径，也不把蓝区契约假定成黄区真实API。

外部环境限制维持：不恢复已过期Docker入口隔断；若必须做33实测，只能通过真实管理入口
对确切且同时匹配 `c23-attention33*`、`dcat.task=17-fault-c2c3-20261117` 的容器申请≤10分钟豁免。
入口尚未取得，不能声称已暂停清理。GMM1缺少匹配op-plugin来源只阻塞routed MoE，
不重搜旧失败来源、不换wheel、不阻塞普通W8A8。

## 本轮验证与发布

公共动作/目录/实际请求CPU契约/单点恢复/源码包组合共204项：203通过，1项因本机
Windows不允许创建源码符号链接而跳过（非NPU测试）。另13项保留的39整组CPU研究契约
通过，显式走研究测试入口，未恢复公共整组路由。合计216通过、1跳过。
首轮组合中的两个新增断言误把 `QUERY_OK` 当 `ACTION_COMPLETED`，以及一个测试模块
重载后的mock对象问题，已修复并在上述完整组合复测通过，未删除故障/恢复负例。

先执行受影响的公共动作、RMS/RoPE/alias、能力查询与路由拒绝测试，再检查
Host/Runtime单点、实际请求模拟、恢复和源码交付闭包。CPU fixture明确标为CPU，
不作为124或实际主项目验收。后续新增65项39/量化合同全部通过；源码包29项中28通过、
1项Windows符号链接权限跳过。Source116已有上述39 NPU实测；量化首轮加载失败0F，
Source117加载/normal通过但reference在admission前拒绝，仍为0F；Source118六阶段及健康实测通过，
28只有无设备构建。普通量化frame修复的67项定向CPU通过；当前请求显示修复的97项组合通过。
没有新增msprobe原始数据或完整张量发布。
GitHub只提交实现、测试和本总结，既有原件和Native库不进入本次提交。
