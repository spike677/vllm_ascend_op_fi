# 黄区codex交接：2026-11-26

目标：A3 / Ascend910_9382 / Qwen3.8 W8A8 / 0.26、CANN9.1 / TP2，经CLI开启后真实用户请求top-5 logprob有变化，关闭后金丝雀逐位恢复，负例被拒且服务存活；正式登记后免许可使用、合入黄区master。本次只上传代码、文档和已测证据，不发新wheel，不改r8，不重启黄区实例。**按新D73，上传后蓝区转待命；Qwen与V4均由黄区负责人/黄区codex自行决策、验证、登记冻结及合入，不等待蓝区常规审查。**

## 取件与版本

- 分支：`codex/yellow-handoff-20261126`；基线为已推送的`53ec934169a534052c01a045dfdd480525da67c6`，包含effects线及后续owner修复。
- r8冻结发行：`e38c4113b421ac979699071a60e80d3babd05e07`，标签`demoncatpfi-rc3-preview-20261126-r8`。r8.1维护分支仍无登记提交。
- 已核对GitHub：effects线`fad902e384176d2e53954ef54277ad22962ad3cf`包含9464df0c（token-ID效果修复）、5f1dc24f（B8无注入对照）、sidecar/N2、r8.1骨架；owner线53ec9341包含CPU和A2复现/验证。提交可达性记录随索引。
- 取本次`YELLOW-DELIVERY-INDEX-20261126-r3.md`所固定内容提交；按完整SHA核对后，由黄区PC传到`/home/haoran/rc3_r72/incoming/`。使用git archive保留Git文件字节，避免Windows换行转换；不要把开发线checkout覆盖r8的SITE。
- sidecar/N2继续使用已发`native-n2-tools-20261126-r1`独立包（附全部依赖）；本次不另发适配器或补充ZIP。效果工具和owner修复均为开发线未冻结，不能混装进r8并沿用原implementation身份。

## 41行与r8的16项

完整41行、A–F归属和逐行progress_state见`docs/worker_core_v1/yellow_handoff_progress_20261126.md/.json`。类别10/17/8/1/2/3，按逐行数据生成；没有一行因本次交接变成REGISTERED。库行号和公共编号不得混用。

|r8项|A3页面状态|产品/登记缺口与处理|
|---|---|---|
|18.api_half|通过|双rank身份草稿齐；等合格负例|
|18.variance_over4|通过|黄区CLI缺变体映射，PRODUCT_RUN_INCOMPLETE；当前不登记|
|18.variance_under4|API_TOP5_REQUIRED|转r8.2；token字符串键碰撞已由A2证明，token-ID请求修复通过；不是A3已通过|
|22.wrong_row|通过|双rank身份草稿齐；等负例|
|39.output_proxy|通过|双rank身份草稿齐；等负例，关联owner控制证据|
|39.matmul_range|通过|双rank身份草稿齐；等负例|
|28.fia.denominator_replay|通过|双rank身份草稿齐；等负例|
|17.spatial_padding|通过|补图片/内部request/owner关联、changed_elements差2原件、视觉负例|
|quant.batch_shared_scale|OFF_API_NOT_EXACT|转r8.2；无注入首请求也不稳定，不把组成相同当充分确定性证明|
|15.alias_detach|通过|双rank身份草稿齐；等负例；B档ND别名断开，不含私有格式条件|
|41.tail_column|通过|Native sidecar按黄区文字已校验；仍需可信配置负例、正对照完整链路和双rank实读身份|
|26.fia_destination|通过|同Native边界，不能凭maps存在即登记|
|20.buffer_substitute|通过|同Native边界，不能复用别组合的库资格|
|external.module_input_gain|通过|外部module输入示例；许可路径，不进公共免许可登记|
|external.dispatch_input_gain|通过|外部dynamic quant参数输入示例；许可路径，不进公共免许可登记|
|external.conv3d_output_gain|通过|外部视觉示例；许可路径，用实际视觉身份c09d46ff…关联后重新提取|

33不在上述16项内，当前关闭；六条采集门槛不等于tiling/ABI/控制重算完成。公共8、库13的V4路线由黄区自行判断，需要修改D包或公共代码才升级。库12原文已在11-24收到，原台账“缺原文”状态过时；仍须黄区验证lm_head权重行变体。

## 卡位、目录和工作边界

|槽|卡|端口|用途|
|---|---|---|---|
|D/G|0–7|19301|v4b-07，r3/G1；噪声与targeted按顺序采，不等待A系列|
|A0|8/9|19105|Native可信配置负例、41正对照与身份|
|A1|10/11|19101|18验收与22注册拒绝、owner/RPC审计、通用负例|
|A2|12/13|19103|本轮并入A1，不占用|
|A3|14/15|19104|17及外部Conv3d视觉路径|

运行根`/home/haoran/rc3_r72/V4/r8-run/`；子目录SITE、SOURCE、RELEASE、EVIDENCE；状态`r8-state.md`；N2副本`n2tools/`。黄区宿主`host/yellow_accept_r8.py`、`yellow_session_daemon_r8.py`、`yellow_worker_ext_r8.py`、`yellow_negatives_r8.py`、`native_neg_19105.py`。黄区原语清单位于其develop分支`docs/agent-memory/PRIMITIVE_REGISTRY.md`。

### D73权责（替换原A/B分工）

|事项|决定、执行、验证与合入方|
|---|---|
|黄区宿主、harness、扩展类、适配器、驱动脚本、CLI、启动参数、卡位端口|黄区|
|负例设计和判定、是否通过、是否重跑|黄区负责人/黄区codex|
|r8.1登记、validate_record自检、只改登记表的提交与冻结|黄区，无蓝区事前/事后复审门槛|
|41行台账、LEDGER-FINAL、master合入、冒烟、手工指南、闭环判断|黄区|
|V4 D/G结果、公共8与库13判断|黄区；需公共代码或D包修改时升级|
|公共运行时、机制、tools验收工具代码、Native重构建、新能力/修复包|蓝区，仅在收到升级单后处理|

黄区只在自己的扩展中记录RPC，不修改冻结公共runtime，不调用私有rollback掩盖问题。r8.1仅登记表JSON是上述公共文件保护的明确例外；不是允许修改公共Python代码。蓝区不直接联系/操作黄区。历史442文件逐字节相同证明对应9464df0c效果工具版本；当前交接分支另含未发布C2 Host候选，不能把交接分支整体安装为r8。

### 黄区自验证门槛

两个owner，每个rank满足eligible=attempted=applied=consumed=restored、failed=0；changed按发生变化的调用计，允许changed<applied，另报changed_elements。消费者、区域字节恢复、OFF精确、金丝雀精确均有原件。负例exit4和拒绝码、引擎存活、所有rank审计零、同一引擎正对照。validate_record通过才登记，不伪造/截断证据，不以历史结果代替r8。

若黄区决定偏离执行单，必须写黄区决定Y-D编号、理由、影响范围，不能默默降低标准；需要放宽公共准入、改公共代码或冻结包时仍走升级单，不能因本地决定越权。

## A1/A0/A3事实与执行顺序

**A1：不把假设当最终归因。** 最新文字称正对照遇到第三个owner；已核对三个源码SHA，支持排除③这些文件版本差异。蓝区冻结r8两次按18验收+22注册拒绝→missing_permit→全rank审计→新owner正对照，均清零并通过，未复现①。若审计有效，零到非零支持②中间又安装；但④审计读错对象尚未排除，因此不能无条件宣布①已排除。不要用这些推断填negative_controls。

1. 黄区先验证审计：已enable期间必须见每rank bindings=1且owner正确，确认实际Worker/model对象，不能读Wrapper的空属性。
2. 在黄区扩展的demoncat_product_binding调用前后写Worker端序号、时间、PID/rank、action/profile、owner/generation、结果；保留所有辅助调用。日志不写鉴权秘密。
3. 同实例按registered22拒绝、missing_permit、立即owner审计、同引擎新owner正对照执行。再做同client already_enabled、repeat_disable、未知profile、fault99。缺许可要exit4、完整回滚、审计零、正对照通过；不能只看recovery.resources_zero。
4. 第二ownerALREADY_INSTALLED被放大为exit5已是蓝区实证公共Host缺陷；r8.2候选仅此精确分支返回exit4，无其他owner rollback；不能把C2修复当作C1来源已查明。

**A0：**已有5类用户RPC负例只证明用户不能选Native路径/参数；wrong_native_sha256返回UNKNOWN_FIELD不能替代可信部署边界。按`r8_native_configure_negative_boundary_20261126.md`构造错误SHA、正确SHA配不存在路径，configure逐rank结构化拒绝且无库映射；再换正确配置新实例，41真实请求逐rank计数→disable→detach→remove_permit→审计零→身份。PID需对应rank；两个PID各四段maps是八段映射，不断言len(maps)==2。sidecar不回填历史null原件。蓝区A2有等价共享边界实测，不能借此完成A3登记。

**A3：**纯文本用户请求不应被强制HTTP4xx，检查17不注入与理由、服务存活；再做合格图片正对照。17两份计数原件按owner、内部request ID、prompt SHA、图片像素SHA和逐请求changed_elements关联；不能把55154强改55152。外部Conv3d必须核对实际视觉身份，不能仅替换一个SHA字符串。

其他已纠正：health=000不是已证明服务故障；令牌未传是黄区configure接线；configure失败立即停该用例，不能继续装许可；remove_permit不可无条件清别人的会话。同容器换项先停旧server并确认health指向新PID。过往A0端口19101冲突已改19105；sidecar身份在正确恢复清理边界重读。

## 蓝区C1–C7状态与交接责任

|编号|截至本次交接|下一步/归属|
|---|---|---|
|C1|A2冻结r8两次未复现；逐rank/RPC/源码证据已推53ec9341|黄区补Worker日志并自行定性；证明公共缺陷才升级|
|C2|8f0c0937最小Host修复；CPU43/43；A2冻结exit5→候选exit4，其他owner不受损|候选已准备；等黄区升级再发修复包，未知/部分成功失败不放宽|
|C3|标准已明确，A2坏SHA/路径配置双rank拒绝且库未加载|黄区完成A3对应负例及Native完整链路|
|C4|under4工具修复、Batch组成协议A2通过；无注入首请求仍存在冷启动差异|待黄区升级索取r8.2才推进；不放宽精确恢复、不深挖第三方kernel|
|C5|17同事件集两汇总路径CPU/蓝区核对无常数差2|黄区自行判断差2对登记的影响并记Y-D决定；要改公共工具才升级|
|C6|N2 launch_host已交付；V4 r3独立运行|黄区本地跑launch_host、判断V4结果；需解码/窗口二/r7.5/D包时升级|
|C7|r8.1七项expected_ranks草稿已齐，negative_controls=null|新D73授权黄区自检、登记、冻结；不等待蓝区复审|

C1–C7是现状/依赖目录，**不再是蓝区主动队列**。只有下述升级单触发蓝区后续工作。C1–C3实证见`r8_owner_rollback_a2_20261126.md`及同名目录JSON。三个窗口约810秒，均撤销隔离。本交接无新增NPU任务。

## r8.1由黄区自检并冻结

1. 黄区从冻结发行e38c4113单独开维护分支；差异只允许`src/demoncat/runtime/vllm_ascend/worker_extension/product_registered_records.json`，保留原BLUE记录，不合并整个交接/效果分支。r8旧SITE、标签、包原样保留。
2. 负例合格后，按`r8_1_registration_draft_smoke_20261126.md`生成7项+Native3项的精确组合；over4若补齐CLI与产品原件可加入。外部三项仍走许可路径，不写公共免许可登记。Batch/under4等待按需升级。
3. 自行核对两owner×两rank八项计数、消费者、字节/资源/金丝雀恢复、top5变化、完整身份与原件SHA；17补图像关联、差2判断写Y-D决定；Native补实际双rank映射、sidecar/CANN及完整链路。缺件不填True或默认值。
4. 每个身份validate_identity、每条记录validate_record；记录SHA与完整证据对应。运行登记CPU自检，再做只改登记表的提交。**黄区负责人决定冻结，不需要蓝区复审或打标签**；不覆盖已有r8标签。保留标签/提交/登记表SHA和本地安装验证收据。
5. 从该SOURCE构建独立新SITE并跑安装检查。JSON登记表不参加implementation_identity，其算法只散列worker_extension及fault_injection的.py文件。实际安装后必须重新计算并确认完整值为`003294ac9e7468b2069719a44f9931514a9ce504c6b12e6c39f64684878f0bee`，不是只比前缀；不一致时停止并查导入路径/源码混装，不能改expected_ranks来迁就安装结果。
6. 按冒烟单做所有已登记项免许可正例和未登记/漂移/跨TP等负例；失败由黄区判定并按D64处理。记录“无验收许可”、精确身份、金丝雀与资源零；不回退许可掩盖注册失败。
7. 黄区备份后增量合入master（r8与黄区r8.1、CLI映射、宿主修复、模板），完成CLI list/enable/status/disable冒烟与手工指南。按闭环4.3自行判断product_closed，写Y-D决定和LEDGER-FINAL。

建议CPU命令（固定版本完整源码树，不只拷一个工具）：

```sh
PYTHONPATH=src:.:tests/worker_extension python -m unittest test_product_registration test_product_registration_pipeline
PYTHONPATH=src:. python tools/product_registration_pipeline.py registration-input.json --output candidate-record.json
```

pipeline自身产REVIEWED_DRAFT_ONLY；黄区完成自验证后才按该工具既有--registry/--approved-sha契约落入候选登记表，不改工具。新版安装身份检查在黄区实际新SITE执行，例如：

```sh
# NEW_SITE必须是此次独立安装的绝对目录，不能指向r8旧SITE。
PYTHONPATH="$NEW_SITE" python - "$NEW_SITE" <<'PY'
import pathlib,sys
from demoncat.runtime.vllm_ascend.worker_extension import product_registration as r
root=pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(r.__file__).resolve().is_relative_to(root), r.__file__
observed=r.implementation_identity()
print(r.__file__); print(observed)
assert observed=='003294ac9e7468b2069719a44f9931514a9ce504c6b12e6c39f64684878f0bee'
PY
```

宿主CLI仍为黄区`python -m demoncat.cli fault ... --socket "$SOCKET"`，不要求使用开发线入口。implementation SHA来自已有七项14份实际身份；它不替代Native和模型等其他身份校验。

## 仅三类情况升级蓝区

1. 有证据证明公共代码缺陷，且不修改公共文件无法解决，例如C2阻挡负例、A1第三个owner最终证明由r8自身错误安装。
2. 需要新能力或新包：under4/Batch的r8.2、33解码与窗口二/r7.5、V4 D包、Native重新构建。
3. D64第5条停下情形且超出黄区权限，例如放宽准入或改冻结包。

升级单≤4KB，包含：一句话问题；最小复现；r8标签/源码清单SHA/实际两rank身份SHA（TP8按实际rank集合，不能只取两rank）；原件文件名、字节、SHA；已经尝试的步骤和按D64哪条作出选择；希望蓝区提供修复、新版本或判断。由负责人和蓝区codex处理并验证；交付后由黄区复验并决定合入。不是每个测试结果都要回蓝区。

## D64执行边界及回传

沿用负责人D64：工具覆盖缺口先保存已完成原件、只停依赖缺口的子任务；单项最多重跑一次且保留两次原件；外部证据未到登记等待，不阻塞独立槽；不能越权修改公共运行时、破坏恢复或扩大批准范围。身份/对象不匹配、资源不清零或需要修改公共代码时停对应实例并向黄区负责人报告，不自行清别人的binding。已有授权内黄区宿主/CLI修复可自行推进，每轮在本地状态与Y-D决定说明所用分支。不要手工编造N2字段。只在里程碑（负例完成、r8.1冻结、master合入、第一版闭环）或升级单时回传，每份≤4KB，文件名/字节/SHA及本轮黄区决定原样保留；不同运行不能拼接。

本次不停止A系列或D/G，黄区按新D73立即继续。新建黄区决策存档Y-D1、Y-D2……，状态沿用r8-state.md。蓝区完成本次上传后待命，不主动推进C1–C7、不做常规复审；仅收到升级单时介入。

## 交付确认

分支：`codex/yellow-handoff-20261126`。内容提交、索引固定内容SHA和远端核对结果在封存后追加；索引按不可变内容提交取件，后续确认尾注不替换索引所指正文。
