# r6 A组补充提示单：金丝雀、许可、profile与并行

由负责人转交openchamber；不修改冻结r6包，不新增黄区组合。本单依据冻结发行`ea42a246`源码、r5.1登记源码及Y1/Y2文本。最新逐页审查见[yellow_y1_y2_review_20261123.json](yellow_y1_y2_review_20261123.json)。

## 1. TP1 Batch：恢复未证明，不能写“已排除残留”

回传显示注入前六轮中第1、3轮就与参照不同，关闭后只有一轮；7904次applied/consumed/restored、failed=0和无Native句柄是有利旁证，不能代替恢复证明。并发调度可能造成批次不同，但未拿到实际每step组成/输入与新进程对照，不能断言全部差异就是BF16合批。两种问题必须分别记录：**非等价金丝雀比较**，以及**尚未排除的状态残留**。

### 修订规范：固定串行B1金丝雀

- 仍保持部署 `max_num_seqs=8`、`max_num_batched_tokens=8192`，故障效果测试仍用真实并发用户请求，不把服务改成max_num_seqs=1。
- 金丝雀用部署侧固定的2～8条不同prompt，按固定顺序逐条HTTP提交；等待一条完整结束并drain后再发下一条。每条固定4token、相同seed/greedy/前缀；观察实际scheduler每步都只有该请求，不仅检查host串行调用。
- 开启前至少**连续3轮**金丝雀的全部hidden/logits有限且逐位相同，每轮覆盖全部固定prompt；最多6轮。任何调度B>1、其他用户请求混入或分块/前缀不同直接判无效，不把它算成一次数值比较成功。
- 关闭后同进程至少**2轮**相同串行金丝雀，两轮均需逐位等于开启前有效参照，且三项恢复同时成立。没有“多跑到碰巧相同”为止的规则。
- 这是恢复诊断/验收的固定输入协议，不证明任意并发用户输出具有batch不变性。故障阶段仍必须验证多请求同decode批次的真实注入，以及B1用户请求不注入负例。

**冻结r6现在不能直接采用此规范跑PASS。** `product_batch.AutomaticBatches.finish`要求每条金丝雀见到多请求decode批次（`decode_batch_seen`）；串行B1会报`PRODUCT_BATCH_CANARY_COMPLETION`。`ProductClient.enable`仅检查末尾连续2轮，`disable`只采1轮。因此这不是给黄区换一个HTTP发送循环就能解决的问题，需r6.1最小维护修复，并同步开发线；不得在黄区改冻结源码或删掉断言。已确认的是工具协议缺口，不是Batch注入内核残留根因。

维护改动限定：Batch金丝雀允许并强制真实串行B1、记录规范化调度轨迹（prompt身份/顺序/computed/scheduled/phase）、3轮基线和2轮关闭后证明、审查器校验该协议；故障量化方法、容差、Q/scale标准、其他故障规则不变。r6.1从`ea42a246`派生，独立CPU和A2实测后新包/新标签；在交付前TP1/TP2 Batch恢复不得按新规范登记。旧回执保留。

### TP1诊断兼复测执行单

1. 保存失败session的status、各rank最后disable/unregister/detach回复、回执/安装/配置SHA；禁止覆盖旧文件。
2. 按下节清理，只在清理完成后复用实例。记录实际patch函数身份、enabled、hook、lease、inflight、Native句柄；未知填null。
3. 对旧证据复核所有实际batch调用（不能将每请求参与次数重复求和）的before/restored字节SHA和消费证明。原始完整数组留黄区，仅传索引/实值摘要和文件SHA。
4. 新维护版安装检查通过后，在同一实例按新协议获得3轮稳定基线，跑两个独立owner的真实并发故障请求，各自关闭后2轮；两owner之间不重启、不重置GDN/KV来制造恢复。
5. 另起相同镜像/权重/参数的无注入对照进程，采同串行金丝雀。保留原失败实例证据；新进程只能帮助定性，不能作为旧owner恢复证据。
6. 各rank每prompt/step回传输入及轨迹SHA、hidden/logits形状/dtype/有限性、差异元素数、首差索引及B/F值、max_abs。轨迹不同→比较无效；轨迹相同且新进程也不稳→环境/基线；只有原进程漂移→残留候选，不直接归因某kernel。

### ownership_retained=true时的清理

先区分**某次disable回复**与**最终当前状态**。Runtime的disable本来不释放owner；成功的unregister/detach才释放。ProductClient可能已完成Runtime清理、但因金丝雀不等而保留产品binding，不能只拿早期disable的一行判断最终租约状态。

1. 停止本实例接收新请求，drain所有在途请求，保存 `client.status()`、`cleanup`和每个既有`WorkerFaultClient.receipt()`；不要新建owner来替代失败owner。
2. 对仍持有attachment/lease的原rank client，按已存在Runtime失败路径执行 `recover()`；核对`ACTION_COMPLETED`及恢复字段，随后`unregister()`、`detach()`。已明确DETACHED的rank不重复操作。缺失回复、device消费未完成或recover失败时保留所有权并隔离实例，不手改内部dict。
3. 所有rank实际资源为零、Runtime已DETACHED后，由可信部署调用已有 `demoncat_product_binding` 的 `detach` 动作移除该profile的连续binding。它仍检查无租约/无执行中的warmup；若断言拒绝，停止复用并保留失败。不要将 `client.recovery['restored']` 强改true来绕过ProductClient.detach。
4. 移除同owner的临时验收许可，读取并核对许可/hook/patch/lease/inflight为零。最后关闭此失败产品session，新的测试使用新owner。该清理可以证明资源释放，**不能改写旧金丝雀恢复失败**。
5. 若无法证明安全清理，专用实例隔离并退出，保留诊断后新起实例；旧验收记失败，不记“重启恢复通过”。

可信RPC示意（已有executor/operator/profile，非普通用户接口）：

```python
# 仅在原Runtime各rank detach成功、无在途请求之后。
rows = executor.collective_rpc(
    'demoncat_product_binding', args=('detach', {'operator': operator, 'profile': profile}))
assert all(r['status'] == 'PRODUCT_BINDING_REMOVED' for r in rows)
# 随后由既有可信部署许可移除流程处理原permit并记录实际结果。
```

## 2. TP2 Batch复核

先回传已生成提取文本，不重起实例。提取v1只含聚合canary_exact，不含实际批次轨迹，因此**不能仅用提取文本判断串行/并发**。需补原宿主发送方法源码SHA、canary prompt数及顺序、是否并发、前后实际step组成摘要（≤4KB）。若采用r6默认多prompt并发金丝雀，原记录保留为旧协议结果，按新维护规范重做两个owner；不得用恰好相等推出batch无关。未知轨迹同样不能授新规范恢复资格。

## 3. 缺许可负例：用合法、已配置的同一profile

源码依据：`deployment_acceptance_report`对continuous/product路径最后使用 `can_prepare=implemented and scoped_permit_valid`，不是“凡有登记都不需要许可”。r5.1 `reviewed_deployments.py`将14项记录在`YELLOW_CONTINUOUS_RECORDS`、`execution_mode=continuous_acceptance`，明确不授历史ordinary/Native路径，仍需作用域许可；冻结r6又是独立发行，不能假定自动包含之后的r5.1登记。r6产品actual还带`execution_mode=product`。因此“已登记就无许可放行”的前提不符合当前发布合同。

使用正常可验收的18或Batch profile，正确硬件/模型/软件/TP/范围；**不构造未登记/错误profile来测缺许可**：

1. 干净专用实例，configure合法部署和profile，保留身份检查；确认没有旧许可/owner。
2. attach并尝试enable；不安装许可。预期prepare失败：外层`PROVIDER_UNAVAILABLE`，报告检查项`deployment_identity`包含`ORDINARY_ROUTE_NOT_ACCEPTED_FOR_DEPLOYMENT`，`acceptance_permit=null`、`can_prepare=false`。`DEPLOYMENT_ACCEPTANCE_SCOPE_CHANGED`不是该负例必须出现的错误。
3. 保留报告中全部实际身份和不满足项；若还缺runner字段、源码/模型不匹配或金丝雀未达标，本次负例不够纯，不能证明只因许可被拒绝。
4. 清理负例产生的binding/attachment；同一部署及profile安装有效作用域许可后验证prepare可通过（正对照）。不需要为负例制造任何故障请求；随后清理许可与资源。

Batch的8192/8与其他项1024/1确为不同验收参数，必须单独记录，但不是本次无许可拒绝的唯一已证根因。r7或后续维护版中只能审查登记**产品模式**组合，键包括TP、硬件/软件/模型制品SHA、mode、参数范围及适用变体；不能把r5.1验收表复制成无条件产品放行。生产无临时许可运行是独立准入变更，需明确实现/负向测试/审查后发布，不在本次提示单中偷偷放宽。

## 4. 未知profile与未知机制是两个测试

|测试|输入|冻结r6实际结果|
|---|---|---|
|类型非法|fault_id字符串`"99"`|`ValueError: fault_id must be a positive integer or quant`；这是参数类型检查，不是profile测试|
|未知机制|整数`99`、variant=`unknown_profile`|FaultConfig的catalog.resolve先报`ValueError: Choose an explicit mechanism variant: ...`；仍未到profile查找|
|未知profile（所需负例）|合法FaultConfig如18/api_half，可信RPC引用未配置profile名|`PRODUCT_PROFILE_UNKNOWN`；不会安装binding或注入|

```python
# 可信专用实例，operator已经授权；不用创建虚构故障号。
executor.collective_rpc('demoncat_product_binding',
    args=('capabilities', {'operator': operator, 'profile': 'not_configured_negative_probe'}))
# 预期抛/返回包装后的 PRODUCT_PROFILE_UNKNOWN；记录每rank错误码、无资源变更。
```

若黄区CLI不暴露profile选择，则由已有可信session/bootstrap运行上述RPC作为profile负例；CLI单独运行整数99未知机制检查。两项分别命名，不能互相替代。CLI将ValueError/契约拒绝映射HTTP400/422等4xx并带稳定错误码；真正服务端异常仍5xx。该HTTP映射属于黄区接线，公共库不伪造HTTP响应。

## 5. 实例级并行与卡组

每个服务实例（含其全部TP/EP ranks）同一时间最多一个公共或黄区自有故障；不同卡上的独立实例可各运行一个。全局互斥的键是服务实例身份，不是全宿主机，也不是单rank。各线必须独立服务端口、HCCL master端口/端口段、会话进程、通信组、owner、回执目录；加载权重错峰，失败只清理本线。不得跨线停止服务或清文件。

负责人已确认0、2、14号卡已释放，A组用0–7，B组用8–15；B组噪声标定、只观测、decode计数及后续D批所有步骤固定使用8–15。该卡状态以负责人声明为依据，不写成蓝区实时观测。TP8初始化失败时回报，不私自降低卡数；仍遵守实例/端口/目录隔离。Codex不向黄区直接派发任务。

## 6. 回传审查与本次交付状态

TP1本地文本实际有12页：18/39op/28/17各2页有效；22仅scope1一页且SHA不匹配；39mr两页中scope0 SHA不匹配；Batch是完整失败提取一页。**不是六项各两页全部有效**。要求原样重贴22两页、39mr scope0；不重算SHA盖过传输错误。提取v1没有owner身份、独立负例证据，仍需原件的owner/作用域关联摘要。TP2已有六项文本直接回传，另补18 scope1/rank1及TP1/TP2有效负向检查。

Y1接线提交`7d509a2b`仅按回传记录，蓝区未读到六文件源码，不宣称代码审查通过；master合入按负责人流程，未替黄区操作。Y6第3/37配置说明和20/28/38映射维持原决定。

故障15只读工具和Native A3执行单另行交付；**15 B档已由负责人批准**，见[档位决策](fault15_tier_decision_20261123.md)，进入阶段2但先补真实自然写入链，链不成立不注入。r7 N批不依赖15完成。此单可以先交负责人；Batch正式复测须等待包含代码修正与验证的新维护件，不让黄区在冻结r6上重复做无效串行试验。
