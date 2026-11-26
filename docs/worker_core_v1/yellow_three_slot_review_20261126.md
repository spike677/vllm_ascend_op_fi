# 黄区三槽重排与最新问题归属

依据本轮负责人附件及 `yellow-a1-fresh-audit-bff9af4b` 原文；身份SHA见 `r8_2_owner_refusal_candidate_20261126.md`。这是文字/源码审查，非codex直接操作87所得结果。

|项目|本轮定性|执行方/缺件|
|---|---|---|
|A0手工启动|已有health200、补全200、RPC200及Paris, France回答，不能说该次引擎未就绪；负例脚本被YELLOW_BOOTSTRAP_TOKEN_REQUIRED拦在configure|黄区核对两端令牌传递，日志只写变量名，configure未通过立即退出；不把鉴权失败算Native负例|
|health000|未取得HTTP响应，不能仅据此定性引擎失败|黄区保留curl退出码与stderr、使用本机直连，核对server.pid及进程身份；代理仅是候选，不是已证实根因|
|A2模型检查失败|缺原始前置traceback，不能仅凭其他槽成功确定设备映射或trust_remote_code是原因|黄区对比真实启动参数与模型路径；PRODUCT_REGISTERED负例并入A1|
|A1 missing_permit|本次新实例初始审计零，残留发生在当前用例期间；调用路径/清理失败原件仍缺|黄区提交直接RPC或ProductClient的实际执行证据；先不使用私有回滚补丁|
|新owner拒绝放大|公共Host路径已CPU复现并准备最小候选|Codex，待同次owner/generation/RPC证据后定稿；见候选说明|
|A3纯文本负例|普通纯文本推理不能因17启用而被要求HTTP4xx|验证故障层不注入/拒绝，而非推理服务必须拒绝文本；若实际做不到，保存证据再定性|

本轮安排为三个槽：A1=10/11、19101；A0=8/9、19105；A3=14/15、19104；12/13不用。该表替代此前四槽负例分工，不改变冻结执行包和正常机制验收范围。黄区自行修改宿主鉴权、共用驱动和启动器，codex不代改黄区组件，不重发冻结r8。

A1先在未装许可时测22的PRODUCT_REGISTERED拒绝，再测18缺许可、未知项、同client重复enable、重复disable、未知profile及fault99；每项保留逐rank前后审计和同引擎正对照。重复bootstrap的ALREADY_BOUND不能不核对实际已绑定部署就任意视为新configure成功。

A0使用已交付d9aa库/c61f sidecar，修正令牌传递后测Native负例和41正对照，再从实际Worker映射及observe_identity取得两个rank身份。本轮仍未获得足够的实读身份与负例以登记三项Native。

A3在17已启用时发纯文本：预期推理HTTP200，所有rank该请求eligible/applied=0并有不注入原因；超范围图同样不注入或控制层结构化拒绝；关闭后全rank审计零，使用合格图做同引擎正对照。看的是请求实际处理后的grid/范围，不能仅凭原文件尺寸猜测预处理结果。若当前运行时给出不同结果，保留它，不由黄区修改公共机制。

每槽最多重跑一次，重跑换新实例；审计不为零只停对应槽并保留现场，其他槽继续。全部有结果并停机确认卡释放后，黄区按既定r3启动D/G。N2是离线事项，不作为D/G前置。

r8.1七项expected_ranks保持已核对状态，negative_controls仍null。文本中的“missing_permit通过”与同用例残留/正对照失败并存时不登记。Native补齐sidecar后的双rank身份、负例及原验收证据后再加入草稿。最新材料没有提供这些完整合格原件，本轮不冻结r8.1。
