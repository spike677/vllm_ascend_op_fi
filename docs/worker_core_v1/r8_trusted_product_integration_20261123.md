# r8 开发：可信 bootstrap、CLI helper 与免许可登记

本页是蓝区开发状态，不属于今晚 r7.1 文档 ZIP，不向黄区下发运行包。负责人已批准免许可登记和 CLI helper；空登记表仍拒绝所有免许可组合，CPU 测试不是 A2/A3 实机资格。

## 官方可信 bootstrap

新增 DemoncatWorkerExtension.demoncat_product_bootstrap(action, payload)。部署在真实 Worker 模型加载后、同一执行线程本地调用 configure_product_bootstrap(worker, source=..., authorize_operator=部署回调, paths=可信只读配置路径列表, canary_tokens=实际processor结果)。该初始化函数不通过 RPC 暴露。它保留已有 authority 的对象/线程/进程绑定，不建立无条件授权。

RPC 参数：configure 仅 operator；permit 仅 operator/mechanisms/attempt/source_manifest_sha256；remove_permit 仅 operator。鉴权操作名分别 product_bootstrap.configure/permit/remove_permit，内部公开 API 继续检查 configure_product_profiles/install_acceptance_permit/remove_acceptance_permit。部署应同时授权相应操作。请求不能携带路径、认证回调、来源、Native 裸地址或令牌映射。身份由部署控制面认证，不能将公网自报 operator 当身份。

所有 rank configure 成功 → 验收模式逐 owner 安装许可 → ProductClient.capabilities（census attach）→ enable →真实 HTTP → status → disable（drain/金丝雀/清理）→ detach → remove_permit。任一 rank 失败时沿现有组恢复流程清理已成功 rank。RPC 不创建/重启 Worker。r7.1 继续使用黄区原包装，不能反向声称 r7.1 已含本 RPC。

## CLI helper / console script

worker wheel 新 console script 为 `dcatpfi-product`，等价 `python -m demoncat.runtime.vllm_ascend.worker_extension.product_cli_entry`。不覆盖黄区 `python -m demoncat.cli` 或 dcatpfi 名称。命令为 fault list/enable/status/disable，--socket 必填，enable 的 --fault-id 必填；没有 rank/layer/value 或许可安装参数。

CLI 参考传输是 Unix socket、单行 UTF-8 JSON，schema=demoncat.product.cli.v1，最多1MiB响应，退出0/4/5。这不是对黄区已有 socket 协议的假设，不能直接替换黄区 CLI。部署自己的 daemon 可调用 TrustedProductSession.dispatch(request, principal=已认证身份)；daemon 负责 socket 权限、peer 认证、服务生命周期和全局互斥。授权回调不可从请求传入。

候选由 ProductClient.capabilities 动态获取，helper 不维护机制白名单。未知机制99拒绝4；跨 rank 候选不一致拒绝；失败 enable 保留 active client 以便恢复。capabilities 已完成 census attach，helper 不重复 attach。旧 fault_injection/capabilities.py 的返回值新增 catalog_scope=LEGACY_RESEARCH_NOT_PRODUCT、product_eligibility_authoritative=false；产品路径无对它的导入。

## PRODUCT_REGISTERED

只匹配随包审查登记：硬件原名和设备、软件版本、模型/量化/所有权重索引SHA、TP/PP/DP/EP、实际源码与运行库指纹、实现指纹、参数范围、模式、变体和Native构建身份。image_digest按已批准规则不是必需身份。EP登记暂未适配，不能借TP2记录授权TP8+EP。world_size=1沿相同匹配路径，只有CPU验证。

明确 PRODUCT_REGISTERED 的 profile 在 capabilities、attach、准入和生命周期控制边界重新匹配。未登记/缺字段/身份漂移/跨TP/范围变化拒绝；若同时存在临时验收许可则拒绝混用，不回退到许可模式。PRODUCT_ACCEPTANCE_ONLY 原逻辑保留，仍须许可。

tools/product_registration_pipeline.py 将原样提取页、实际完整身份、owner表、三项负例及正对照整理为 REVIEWED_DRAFT_ONLY；核验提取SHA、审查器SHA、八计数、scope×rank完整矩阵、消费者和恢复结论。登记暂存需要明确 approved_sha，不直接 push/tag 或修改运行中的表。黄区原件保留黄区，登记保留 YELLOW_TEXT_ATTESTED，不改写成蓝区原件审查。

当前 product_registered_records.json 为空。还须在新实现上完成 A2 两 owner 的验收证据审查与无许可正例，之后才可加入精确 BLUE_COMBINATION；r7.1 历史回归不能自动授权 r8 代码。

## 发行和剩余门禁

构建器将 docs/ 放在发行目录可直接读取，同时保留源码包中副本，纳入release清单与SHA。冻结 r7/r7.1 不修改。尚需全量 CPU、r8 实机免许可正例、安装/console入口/源码一致性、r8对r7.1差异审查。r8尚未冻结，不发黄区。
