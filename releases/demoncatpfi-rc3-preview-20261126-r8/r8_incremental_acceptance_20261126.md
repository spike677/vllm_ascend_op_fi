# r8 黄区增量检查单（冻结前草稿）

本单由负责人转交；不停止正在进行的 r7.4.1 正式测试。目标是 A3、Qwen3.8 现役 W8A8、0.26/CANN 9.1、TP2。A2 社区权重的回归不能代替该组合的登记。最终版本、源码清单、wheel、登记 SHA、关闭项以 r8 冻结索引为准；本草稿不授予任何资格。

## 先核对登记，再开始免许可检查

每个能力、命名变体分别核对登记。登记键包括硬件原名及各 rank 身份、TP/PP/DP/EP、软件和实际运行库身份、模型 config/量化描述/全部权重索引 SHA、参数范围、模式、变体、实现指纹及适用的 Native 身份。`image_digest` 未观测不伪造；不能把镜像标签当作 digest。

黄区在本地部署配置中使用 `mode="PRODUCT_REGISTERED"`、`source="YELLOW_DEDICATED_INSTANCE"`，保留真实 model 路径、TP2、该版本 source_manifest 及 SHA、命名变体、金丝雀配置。Native 三项还需 `native_library` 和 `native_sha256`，绑定本次审查的 A3/CANN9.1 产物。只改 mode 不能创建登记；未登记组合必须拒绝。

可信 bootstrap 配置后的 `required_permits` 应为空。每个 rank 都先核对没有临时验收许可；本次不得安装、借用或保留上轮许可。黄区会话 daemon 持有 ProductClient 与鉴权回调，不能把 bootstrap 暴露成普通用户请求。能力以实际 `ProductClient.capabilities()` 和 `product_contract` 为准。免许可模式的成功记录必须明确 `PRODUCT_REGISTERED` 和 `product_qualified=true`，不能用验收模式的成功顶替。

当前外部 provider v1/v2 仍走受控验收许可路径。外部适配器的黄区测试和登记由黄区负责；不能把公共机制的免许可登记借给外部适配器，也不能据外部三个 A2 示例宣布全部外部能力免许可。

## 每项的宿主顺序

下表给出 `MECHANISM` 的准确取值。只执行最终登记清单已放行的行；这是命令映射，不是资格清单。18 的三个变体使用三个独立登记键。

|能力|`--fault-id`|
|---|---|
|18 基础档|`18.api_half`|
|18 方差放大档|`18.variance_over4`|
|18 方差缩小档|`18.variance_under4`|
|22|`22.wrong_row`|
|39op|`39.output_proxy`|
|39mr|`39.matmul_range`|
|28|`28.fia.denominator_replay`|
|17|`17.spatial_padding`|
|Batch|`quant.batch_shared_scale`|
|15|`15.alias_detach`|
|41|`41.tail_column`|
|26fia|`26.fia_destination`|
|20|`20.buffer_substitute`|

同一服务不重启，两个 owner 不复用同一会话。按环境模板确认旧 API/Worker 已退出，新健康检查确实对应新进程。使用黄区已有 CLI，命令名不改：

```bash
python -m demoncat.cli fault list --socket "$SOCKET"
python -m demoncat.cli fault enable --socket "$SOCKET" --fault-id "$MECHANISM"
# 发送本项已登记范围内的真实 HTTP 用户请求；宿主不逐请求注册。
python -m demoncat.cli fault status --socket "$SOCKET"
python -m demoncat.cli fault disable --socket "$SOCKET"
```

`SOCKET` 是黄区宿主已建立的可信 session，不由 CLI 创建。list/attach、enable、status、drain→disable→unregister→detach 的结果全部保存；只看 CLI exit0 不算通过。disable 后运行部署金丝雀，逐 rank 确认区域字节恢复、租约/owner/binding/wrapper/在途请求清零，以及精确金丝雀。`recovery_required=true` 时停止该实例接收请求，保存控制记录；不盲删外部 wrapper，不继续切 owner。按 Y1 安全恢复说明处置，必要时只重启该专用 Worker 组。

普通请求固定生成4 token，覆盖 prefill 和 decode；只支持开启期间的全部合格请求。用户参数不暴露 rank、layer、value。超出已登记范围的请求不注入并记录原因。API 请求附 `logprobs=5`（chat 使用 `logprobs=true, top_logprobs=5`），固定 temperature/seed/prompt，将 normal/fault/off 的 top-5 logprob 逐项保留；这不是完整 logits。token/文本必须变化不作为门禁，耗时倍数只记录。

|项|r8 的最小增量检查|双 owner 与旧结果|
|---|---|---|
|18.api_half、18.variance_over4、18.variance_under4|每个已登记变体分别 enable→真实请求→status→disable→金丝雀，不互借变体登记|字节和影响审查通过后沿用各自 r7.4.1 正式双 owner 结果；未通过的变体关闭|
|22、39mr、28、15、20|每项分别完成免许可冒烟和恢复；15 仍为 ND 视图别名断开语义|满足沿用条件的结果保留原版本身份，不改旧回执|
|17|真实图片请求；内置确定性金丝雀；TP2 设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn`|图片路径不能用纯文本请求代替；沿用条件同上|
|Batch|max_num_seqs=8，多个独立 HTTP 请求形成真实共享 decode 批次；另做 B1 不注入负例|金丝雀列表逐条串行发送。若有公共修复，两个 owner 全部重跑；未定性/缺证不登记|
|39op|同组 Worker 上 scope0 完整结束，再创建 scope1；逐次核对 owner、调用序号、wrapper 归属|两 owner 必跑；不以不同层次的 owner 字符串应相等作为判断依据|
|41|先确认 CLI/harness 确实枚举并执行该项，再核对 Native/CANN 身份|两 owner 必跑，旧轮未执行不能当通过|
|26fia|configure 成功、Native 路径/SHA/运行库通过，再做真实请求|两 owner 必跑；只接受同一次运行的配置、效果和恢复证据|
|外部三个示例|module input、DynamicQuant dispatch input、visual Conv3d，按各自许可与适配器身份做增量检查|本轮不将其标成公共免许可能力；黄区适配器修改/正式验收由黄区完成|

受修复影响的其它项若在最终影响分析中新增，随最终清单追加双 owner 重跑。未受影响项也必须有 r7.4.1 正式结果审查通过且登记，才可沿用双 owner 证据。

## 负例与随后正对照

在独立 session 执行，不污染正常已登记配置：

1. 未登记组合或命名变体：enable 返回结构化拒绝，CLI exit4，包含拒绝码；不安装许可来让它通过。
2. 身份漂移、跨 TP 借用、任一必需字段缺失：在身份夹具/独立配置上验证拒绝；不修改真实模型文件或运行库来制造负例。CPU 负例可作为这些路径的补充，不能冒充 A3 实测。
3. 未知 profile、未知 fault 99：CLI exit4，保留拒绝码；随后 status/disable 可完成，引擎健康、资源零。
4. 受控验收路径的缺许可负例仍应拒绝；在 PRODUCT_REGISTERED 且身份匹配的组合上“没有临时许可”正是正例，不能要求它拒绝。
5. Batch 非串行金丝雀应拒绝且引擎存活；改回逐条串行后同进程正对照成功。B1 用户请求正常完成、注入数为0且有原因。

每项负例后都要同引擎运行健康请求和合法正对照。直接 RPC 的 HTTP200 加结构化拒绝 body 可接受；CLI exit5、引擎退出、资源残留不算契约负例通过。

## 四个槽并行安排

仅使用负责人已划给 Qwen A 组的0–7卡；B组8–15卡保留给当前工作，不抢占。黄区确认实际设备空闲后可按下表分配，物理卡分配不等于登记身份可随意替换；每槽必须匹配自己的登记和 rank 身份。

|槽|建议卡|串行项目|隔离要求|
|---|---|---|---|
|A0|0、1|39op → 41 → 26fia → 20|Native/owner 异常项；单次只开一项，逐项核对新API和Worker|
|A1|2、3|18 三个变体 → 22 → 15|每变体独立session；不得复用前项许可|
|A2|4、5|Batch → 39mr → 28|Batch 独立 max_num_seqs=8；切项重新检查调度配置|
|A3|6、7|17 → 外部 visual Conv3d → module input → dispatch input|图片路径与纯文本路径按真实profile切换；外部保留自身验收许可|

每槽独立容器/API端口/socket/输出目录/owner；两个 owner 必须使用同一组 Worker，不为第二 owner 重启。涉及配置变更的切项按照 [环境模板](r8_environment_template.md) 关闭旧实例。无法独立满足内存和设备条件时减少并发，不共享 Worker 或 Native 状态。

## 回传与失败截止

沿用文本提取协议，完整原件留黄区。逐组合回传两个 owner、每 rank 八项计数、消费者、区域字节恢复、资源清零、精确金丝雀、top-5 logprob 比较、负例/随后正对照、完整身份、原件文件名/字节数/SHA、审查器和提取文本 SHA。`changed < applied` 且无失败可通过；必须解释 applied 与 changed 的不同含义。不同 rank 的 owner/调用序号/内部请求归属交叉核对，不能仅看计数相等。

2026-11-26 08:00（北京时间）仍不能定性的39op/Batch，在本次 r8 里关闭并注明原因；不为赶09:00冻结猜测修复或登记。33、V4/D13/公共8继续各自证据门禁，不阻塞本次 Qwen r8。冻结索引必须把“待证关闭”“审查通过已登记”“本轮A2回归通过”分开列出。
