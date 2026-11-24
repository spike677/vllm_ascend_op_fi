# 外部 provider v2：持久区域恢复设计（已批准开发）

2026-11-24最新负责人决定已批准开发，覆盖此前“只设计”。区域原语见external_persistent_regions.py，14项CPU用例通过；尚未接入实际模型、Ingress排空与组事务，未做A2示例验收。v1继续拒绝 EXTERNAL_V1_PERSISTENT_WEIGHT_OR_KV_RESTORE_UNPROVEN，不能用新原语绕过。
复用现有Runtime、owner/generation、RegionLeases、request_plan、产品attach/enable/status/disable/detach及多rank控制；不另建状态机。
本阶段持久区域先在A2验证，再以A3 Qwen3.8 TP2验收；DeepSeek专属条目走独立D包。以下A2试验已经获准，但尚未完成。

## 区域和证明对象

持久权重包含原始int8存储、weight_scale、共享storage/非零offset视图、NZ转换、转置、预打包及设备副本。
不能仅按模块名或data_ptr建立租约。使用活跃对象引用、allocation generation、device、storage identity、字节范围、dtype/layout及绑定源码身份。
模型卸载、reload、重新量化、设备迁移使generation失效；旧备份不得回写新对象。
KV按cache group/layer/rank、request generation、block/page allocation generation、slot和实际字节布局建账，包含MLA latent/rope分区。
页地址、slot index相同不代表仍属于同一请求。页共享、prefix cache、KV connector/offload、未知异步写入第一版拒绝。

## 持久权重协议

1. enable前停止接收新请求并排空所有旧读者、预取与副本更新；每rank核验同一owner/generation及登记身份。
2. 对全部基础和派生存储取得独占区间租约；与其他故障、训练、在线reload及模型更新互斥。副本模块使用一致规则，分片各改本rank片段。
3. 备份每个物理字节区间一次。别名重叠合并，冲突拒绝。预算不足在写前整组拒绝，不部分启用。
4. 通过适配器枚举所有真实消费者及派生副本生成路径；若存在无法枚举的预打包/缓存则不适用。
   基础权重改变后，按官方可证明的失效/重建边界更新派生缓存；没有该边界就拒绝，不能只改原始weight然后宣称生效。
5. 全rank prepare通过后才进入enable写入。设备事件证明写完后放行请求；条件性算子内部不插入collective。
   **开启期间所有请求**读取修改权重，不能声称只影响某个已观察的请求。用户不指定rank。
开启期间保持同一个已注入权重版本，所有真实读取持续受影响；不能每次读都重复乘scale形成未定义的指数漂移，也不能把读取次数伪装成applied次数。
   与v1瞬态门禁有一处必须明确的契约区别：持久权重不能在超范围请求上简单“NOT_INJECTED”，因为它仍会读污染权重。
   第一版拟仅支持可信请求准入能在入计算前拒绝超范围请求的专用实例；或由部署方把它路由至独立干净实例。无法证明这个入口的部署拒绝持久模式。
   不在同一已污染模型上并发执行所谓正常请求，不用逐请求临时恢复来制造竞态。该契约已获负责人批准，真实入口接线仍待实现。
6. disable停止新请求；等所有设备读者、异步副本、预取事件完成，再在同一allocation generation下恢复字节。
   恢复基础和派生缓存，以原备份恢复或经过精确对照的重建；重建不能保证字节一致时不能宣称区域字节恢复。
7. 全rank验证区间哈希、租约零；对每条消费路径重跑部署金丝雀并核对hidden/logits精确恢复。
   只有全部通过才解除暂停。超时或身份变化保留恢复责任和拒绝服务状态，不能清空账本冒充完成。

## KV协议

- 写入前核验真实请求→cache group→物理页→slot映射；错槽位只允许在本故障请求拥有的页中，其他请求、空闲页或不明所有者拒绝。
- 故障写入后，同一请求后续token读到污染KV是预期现象。不能在每次函数返回后立即恢复KV，否则改变了故障语义。
- KV写前key/value临时区可短租约：备份→改写→cache写入设备完成→恢复临时输入；污染已经进入KV，仍需后续请求/页生命周期处理。
- 请求结束或取消：先等待已派发的读写完成。可采用重建干净缓存或独占页清零后交还分配器；记录具体策略。
- 旧快照只可回写仍为同一request/page generation且从未发生后续合法写入的区间；写版本变化后禁止旧快照覆盖。
- 页回收/复用前必须清除污染并取得设备事件证明。已复用页绝不能回写；出现这种竞争应停止该cache group继续服务，按部署受控重建处理。
- KV恢复分开报告：临时输入的字节恢复、持久页的安全重建/清理、故障资源零、后续新请求金丝雀。
  页重建不是原字节恢复，不能把二者布尔字段混用；拟新增恢复类型字段由审查器明确判断。
- prefix cache必须关闭。原故障请求的token轨迹无需回到未注入状态；干净新请求必须通过金丝雀。

## 计数：八项加元素数

以实际可识别的区域变更事件计数，各rank分别汇总，不把每个读者冒充再次写入：

|字段|持久区域定义|
|---|---|
|observed|实际发现的候选变更边界；读取次数另报reader_observations|
|eligible|身份、区间、归属、预算、恢复条件全部成立的事件|
|attempted|eligible中开始备份/变更事务的事件|
|applied|设备确认写入完成的事件；写值相同也计入|
|changed|applied中实际发生数值/位变化的事件，≤applied|
|consumed|已证明至少一个真实设备消费者读取该事件版本的事件；reader/request覆盖另表|
|restored|该事件所有字节租约恢复或已按批准的KV恢复类型安全终结；必须附restore_kind，不混淆字节/重建|
|failed|eligible后备份/写入/消费/恢复任一失败的事件，不允许归为跳过|
|changed_elements|按事件逻辑元素比较；另列changed_bytes，重复写事件单独计，不当全模型去重元素数|

持久权重enable可能只有1次applied、多个请求消费，因此不要求每请求重复applied。consumed最终必须覆盖每次applied，未消费的故障不算通过。
KV多次写同区间分别有版本；恢复只对仍归属且允许终结的版本记录，不能多次回写旧字节。
覆盖缺口、未适用原因、读者总数、异步事件数、未终结页和租约必须附表；审查器不能机械套瞬态逐调用相等式。

## 四项适配判定（拟议）

|能力|能否纳入v2设计|限制|
|---|---|---|
|weight/faulty_requant|有条件支持int8原权重与配套scale一致变更|全部NZ/预打包副本可发现且能恢复；不能仅改被弃用的原weight|
|weight/param_perturb|有条件支持实际被QMM读取的weight_scale|广播/共享scale别名和融合副本均可核验；未知缓存拒绝|
|kv_update_misdirect|有条件支持同请求私有页内的错误slot|不跨请求、不写闲置/复用页；物理layout与消费者映射必须证明|
|kv_rope_quant_loss|有条件支持写入前key/value改写并跟踪KV污染|临时输入恢复不能替代持久KV安全结束；rope/latent布局需逐分支适配|

## 验证与失败路径

批准后的CPU测试：对象地址复用、别名重叠、预算不足、未知派生副本、并发reload、页跨请求/复用、取消时异步事件未完成、rank部分失败、prefix cache开启。
A2/0.26/Qwen3.8 TP2：每机制同组Worker两个owner；先正常稳定基线→开启→至少两条不同请求及多token消费者→关闭→金丝雀；负例分别拒绝且恢复责任保留。
量化权重同时检查真实QMM输入，KV检查下一token实际cache读取，不要求生成token变化。金丝雀覆盖所有受影响消费者而非单一最终token。
故障数值参数固定命名变体，部署方审核源码SHA/身份/预算。A2不授A3资格；V4/8暂缓，不安排其NPU试验。

任一rank prepare失败整组不写；enable部分写入失败时各rank只回滚自己确认的generation，再在必达控制边界汇总。
请求中断仍先排空异步读写，保留owner恢复权。某rank失联/设备错误不能确认恢复，则整组隔离并通过部署方重建Worker/model/cache，禁止宣称恢复通过。
不自动重新加载全模型清除其他owner；这种全局恢复只能在全实例独占且部署授权下进行。
风险是备份显存、全局排空时延、不可观测派生副本、page ABA及错误回滚；均通过拒绝门禁控制，不能靠延时sleep假设设备完成。

最新实施批次：r7.4新增v2；第一步复用RegionLeases实现持久对象/页面代际和恢复原语，第二步接入现有Runtime及可信宿主排空与全rank提交边界，第三步真实weight_scale与KV写入适配器A2 TP2两个owner。共享请求路径若必须修改，逐文件列出受影响复测项，不能宣称“仅新增”来免除复测。当前只有第一步CPU原语证据，**没有v2产品或NPU通过结论**。
