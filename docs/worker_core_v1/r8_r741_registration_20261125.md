# r8：r7.4.1 基线上的免许可登记迁移（2026-11-25）

2026-11-26补记：下文首轮记录保留为历史。最终候选的全项回归、1360项CPU门禁、三个wheel的406文件安装检查和规范源码身份重验，见[当前记录](r8_full_regression_20261125/README.md)。旧实现指纹不自动沿用；登记已依据安装后双owner原件刷新，免许可最终复验已通过，见[双owner正例和未登记负例](r8_full_regression_20261125/r8canon1125r1-review.json)。

本轮开发基于冻结发行提交 `59f6aea5`，分支为 `codex/rc3-r8-r741-20261125`。没有合并旧 r8 分支的机制、新模型或外部适配器改动；没有改动任何冻结标签和发行包。本文件记录开发验收，不是 r8 最终发行声明。

## 改动与字节边界

迁入精确组合登记、可信 bootstrap、宿主 CLI helper；验收模式仍需要临时许可，已审查并匹配的产品组合可以使用 `PRODUCT_REGISTERED`。没有公开的登记写入或许可绕过接口。

登记键核对硬件与设备、TP/rank/PP/DP/EP、软件版本、实际映射运行库、模型 config/量化描述/全部权重索引 SHA、运行参数、请求范围、变体、实现指纹，以及适用时的 Native 构建身份。缺字段、身份漂移、跨 TP 借用和未登记组合均拒绝。当前不开放 EP 登记；外部 provider 保持验收许可路径。

[逐文件对照](r8_r741_shared_byte_comparison_20261125.json)覆盖481个原始运行时文件：474个逐字节相同，7个仅涉及通用准入/控制的文件变化。全部机制文件保持原字节，包括 `continuous_binding.py`、外部模型/采样/视觉 site 和 Native 实现。每个原始文件都列出前后 SHA；新增文件不等于自动具备资格。

七个文件为 `deployment_acceptance.py`、`deployment_authorization.py`、`extension.py`、`product_binding.py`、`product_config.py`、`product_host.py`、`providers/product.py`。保留 r7.4.1 的18方差档、视觉路由和每次控制时的外部 site 校验。

黄区 r7.4.1 机制测试证据可以保留，但新登记/控制路径仍须增量检查；不能改写旧回执的源码身份，也不能用本轮 A2 正例自动登记 A3。

## CPU 与安装检查

- [全量 CPU 门禁](r8_r741_cpu_20261125.json)：1354项，1352通过，2项既有废弃研究入口带理由跳过；无失败或错误。
- 迁移影响项另测53项、最终外部路由/身份/字节检查31项通过；这些与全量门禁重叠，不重复累计。
- 实机发现的负例脚本问题补了实际 `ProductClient` CPU 回归：先 capabilities，再在 enable 检查拒绝及全部 rank 回滚。与字节审查工具合测5项通过。
- [无设备私有安装检查](r8_r741_private_install_20261125.json)：Worker wheel、安装目录及源码中168个 Python/JSON 文件逐字节相同，包含登记表和 `dcatpfi-product` 入口。检查环境为 Windows CPU；不是124安装，也不是最终三 wheel 发行门禁。

首次 CPU 迁移失败原因保留在门禁报告中：旧测试依赖历史 BLUE 登记、旧 RPC 白名单和未迁入的静态目录改造。历史登记只放在测试夹具中，未作为当前组合的放行依据。

## A2 实机范围与证据

组合：Ascend910B4（A2），vLLM0.26 / vllm-ascend0.26rc1 / CANN9.1，社区 Qwen3.8-27B-W8A8，TP2，物理卡2/5，eager。只验收代表项 `18.api_half` 的登记及产品控制链路；不声称其他能力已完成免许可实机验收。

第一轮先在验收许可模式完成真实 HTTP 请求、同组 Worker 两个 owner、全部 rank 消费者、区域恢复、资源清零、精确金丝雀，以及缺许可、未知 profile、未知机制负例和拒绝后的引擎存活检查。

- [独立复核](r8_r741_a2_20261125/r8b1125a1-review.json)：通过，291.375秒（包含启动和退出）。
- [BLUE 登记审查](r8_r741_a2_20261125/blue_registration_review.json)：只登记此组合的 `18.api_half`。
- 登记 SHA：`d457750ea1084d09b7bd2562222c80ec94402ab1e94ed2735fd91ae43f76df5f`。
- 实现 SHA：`54ed5922bbdf09a362d0d01579e7f8333ff6a2be3caf60d820d7717d9a2c1fce`。

第二次尝试未通过，原始失败记录保留：[r8b1125r1](r8_r741_a2_20261125/r8b1125r1-failed.json)。脚本在 capabilities 已建立只读连接后再次 attach，误把重复 attach 当作准入负例；正确准入边界是 enable。只修脚本和回归测试，不改运行时、机制或已审查登记身份。这247.262秒不计为验收通过。

免许可重跑[独立复核通过](r8_r741_a2_20261125/r8b1125r2-review.json)，耗时286.965秒。同组 Worker 两个 owner 均未安装或使用临时许可，capabilities 和 enable 均报告 `PRODUCT_REGISTERED`。每个 owner、每个 rank 的三个范围内请求合计：observed/eligible/attempted/applied/consumed/restored 各1932，changed=1931，failed=0；另一个超范围请求没有注入。changed少一次是写入值相同，不是失败。

真实消费者包括 `npu_dynamic_quant`、`unified_attention_with_output` 和 `aten::index`。两个 owner 的区域恢复、注入资源清零、精确金丝雀均通过。未登记的22在 enable 返回 exit4 / `PRODUCT_COMBINATION_NOT_REGISTERED`，回滚后租约、owner、binding、wrapper均为0，同引擎普通推理继续成功。CLI handler 的未知机制和未鉴权负例也返回exit4；本轮没有验收黄区CLI或socket传输接线。

原始回执及完整数组留在仓库外/ignored证据目录，Git只收审查结果及原件字节数和SHA。退出时出现 Python resource_tracker 的信号量/共享内存清理告警；这不等于注入租约残留，也不能据此宣称宿主所有IPC资源均已独立核验清零。

## 抢占与隔离记录

[完整操作审计](r8_r741_a2_20261125/isolation-audit.json)列出全部停止容器的名称、镜像、绝对时间及恢复动作。本次授权上限为03:30:58–04:15:58 UTC（北京时间11:30:58–12:15:58），只使用卡2/5的有界ATK stop监控。

| 尝试 | UTC开始 | UTC结束 | 结果 | 窗口用时（含模型启动） |
|---|---|---|---|---|
| r8b1125a1 | 03:30:58 | 03:35:49 | 验收许可正负例通过 | 291.375秒 |
| r8b1125r1 | 03:38:19 | 03:42:27 | 脚本检查边界错误，未计通过 | 247.262秒 |
| r8b1125r2 | 03:43:54 | 03:48:41 | 免许可正负例通过 | 286.965秒 |

累计窗口用时825.601秒，约13.76分钟，非纯kernel运行时间。共停止8次仅占卡2/5的ATK容器实例；未删除任何容器、镜像或数据。每次尝试结束均停止并join监控，未改派发器配置、Docker入口、驱动或CANN；末次只读检查确认本轮控制进程和测试容器均已退出。派发能力已恢复，剩余授权时间不继续占用。

## 后续边界

2026-11-25后续任务调整交付顺序：r8优先，33/r7.5与D包作为第二次增量；不再让r8等待它们全部完成。阶段1缺陷候选和A3登记需退补项见 [Y1维护记录](r8_y1_defect_review_20261125.md)。

33和V4仍待有效取证材料，数据到齐后继续各自门禁。没有新增适配器补充包，没有暂停或操作黄区当前测试，也没有向黄区发送内容。A3组合登记需实际正式证据；r8最终冻结仍等待剩余集成和交付门禁。
