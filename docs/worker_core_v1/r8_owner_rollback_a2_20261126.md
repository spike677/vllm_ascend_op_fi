# A1回滚矛盾、第二owner拒绝与Native配置边界：A2复现

组合：124 / Ascend910B4（A2）/ vLLM0.26.0、vllm-ascend0.26.0rc1、CANN9.1.0 / 社区Qwen3.8-27B-W8A8 / TP2、mp executor、卡2/5。不是A3资格。

## C1：冻结r8未复现，不能直接归因为回滚漏清理

使用冻结发行提交 `e38c4113b421ac979699071a60e80d3babd05e07` 的运行时，单独覆盖诊断文件 `experiments/worker_core_v11_20261118/owner_rollback_reproduction.py`。探针只提供可信测试configure、只读审计和调用日志，不改binding或机制实现，不调用私有rollback进行补救。实际Host及两rank的product_host、product_binding、continuous_binding路径/SHA均与打包清单核对。

在同一Worker组上依次执行：

1. 同时configure 18.api_half（PRODUCT_ACCEPTANCE_ONLY）与22.wrong_row（PRODUCT_REGISTERED）。
2. 22未登记拒绝，exit4 / PRODUCT_COMBINATION_NOT_REGISTERED；立即审计全零，再公开disable/detach。
3. 不装许可，用新ProductClient执行18 enable：exit4 / PRODUCT_ALL_RANK_PREPARE_FAILED / ORDINARY_ROUTE_NOT_ACCEPTED_FOR_DEPLOYMENT。
4. **enable返回后、额外cleanup之前**读取两个rank原始binding字典：均为空；Runtime owner、lease、runner包装均为0。cleanup_evidence的两个binding rollback均对应本次owner。随后公开disable/detach仍为零。
5. 安装蓝区验收许可，新owner正对照enable成功；实际4-token请求中两个rank均eligible/applied/changed/consumed/restored=644、failed=0，字节/资源/金丝雀恢复均通过，detach/remove_permit后全零。

冻结版两次独立启动（r8owner1126a1、r8owner1126a2）均完成上述顺序，未复现“回滚成功但原binding留下”。原件摘要、owner、RPC数量、源码清单SHA、原件SHA和窗口记录位于同名目录的逐次JSON。第二次同时精确复现C2，见下节。

黄区最新文字原件：`D:/CodexArchives/20261126/yellow-a1-rollback-originals-a0-gaps/A1回滚原件与A0待补项_r3独立运行记录.txt`，8113字节，SHA `8d3d75703eff25e877a6fb1423b56e11deeca6c75ab5d488334079c13e519ad7`。

该文字同时报告“回滚成功”“结束审计全零”“随后已有binding”，**不是已观测到current_binding返回None的证据**。其三个源码SHA支持排除这些文件字节版本不一致，但不能证明审计对象、导入类身份、全部模块和中间调用一致。

请负责人转交需补的最小证据（优先已有原件；没有才新实例单用例定位）：

- missing_permit enable完整返回、recovery和cleanup_evidence；逐rank关联实际owner/generation。
- enable返回后立即读取实际Worker上的原始binding字典，附owner/generation、Worker/PID/rank、worker.model_runner.model对象身份、binding.worker/model是否同一对象、binding类型与current_binding结果。空字典原样记录，不填假owner。
- 从configure、22拒绝到18负例再到正对照之间的完整RPC时间、方法、action/profile、owner及逐rank返回；包括辅助函数发出的调用。不要往公共payload添加未支持字段。
- 对应同一次运行的实际加载文件路径/SHA。若审计零后有新的attach，按调用序列定位；若原字典仍有本owner且rollback已报成功，再定为公共清理缺陷。

在上述矛盾未闭合前，该黄区missing_permit不作为r8.1 negative_controls。蓝区未复现也不能替代黄区负例或证明黄区驱动有错。r8.1登记草稿保持关闭。

## C2：区分准确分支与更早的包装守卫

第一轮r8owner1126a1在原owner仍启用时创建竞争owner，实际先触发 `V026_UNREVIEWED_ACTUAL_FUNCTION:.../product_observer.py`，再被owner mismatch放大为exit5。该记录保留，但**不是ALREADY_INSTALLED修复的验证**，不扩大候选白名单。

第二轮r8owner1126a2使用自然的控制边界：原owner成功enable、真实请求、disable完成恢复，但尚未detach。此时原owner合法持有binding，而Runtime租约和包装已释放。新owner的attach到达 `CONTINUOUS_BINDING_ALREADY_INSTALLED`；冻结r8回滚它不拥有的binding，报 `PRODUCT_ROLLBACK_OWNER_MISMATCH`，最终exit5。竞争前后原owner的逐rank审计完全相同。

候选仅改product_host.py：在完整rank集合全部精确拒绝ALREADY_INSTALLED、且本owner账本明确没有attachment/ownership时，记本次enable无资源可清理，不向Worker发rollback，保留原拒绝和exit4。`resources_zero`限定CURRENT_ENABLE_ATTEMPT，global_resources_zero=null；不能借此宣称别的owner资源不存在。部分成功、不完整回包、未知错误、attach后prepare失败仍走原清理。机制文件及Worker owner保护不变。

候选 `2191a50393c63a8b796c5f2161d7c0b7d3fad6fb`（Host修复来自8f0c0937）在r82owner1126a1使用与冻结版第二轮**相同的诊断源码**完成A2验证：两个rank都报ALREADY_INSTALLED，最终exit4；竞争owner的binding rollback RPC为0，原owner审计逐字段不变，原owner正常detach，最终全零。C1缺许可回滚、新owner真实请求、恢复、两项Native配置负例同时通过。该结果只覆盖本次精确拒绝分支；第一轮观察到的更早包装身份守卫拒绝未纳入本候选，不能宣称任意attach拒绝都已处理。

此文不是发行公告。相关7个CPU模块43/43通过。全product CPU集合在Windows导入torch时被c10.dll/WinError126阻塞，不记通过。差异证据见 `r8_owner_rollback_a2_20261126/runtime-delta.json`：428个运行时文件中只有product_host.py改变；258个fault_injection文件逐字节不变。

## C3：坏可信部署文件满足测试设计，实际证据限定共享边界

两种configure负例实际分别返回：错误SHA → PRODUCT_NATIVE_REVIEWED_LIBRARY_REQUIRED；正确审查SHA配不存在路径 → PRODUCT_NATIVE_LIBRARY_HASH_MISMATCH。两rank均结构化exit4，拒绝前后无libdcat_postwrite映射，资源审计零，之后仍能真实推理。

本轮是在18/22诊断完成、目标Native库从未加载的同一实例上测试，没有把它说成A3全新实例验收。可复用为冻结r8共享文件验证函数的A2回归；不能替代A3的正确配置实读库身份、41/26fia/20实际消费与恢复。具体执行与复用边界见 `r8_native_configure_negative_boundary_20261126.md`。

## 隔离、原件与交付边界

每个窗口只停止设备映射完全落在2/5的非本项目容器，记录名称、镜像、绝对时间。未删除容器或数据，未改Docker入口、驱动、CANN或调度配置；窗口结束撤销并等待有界监控退出。逐次JSON列出实际停止记录、起止时间和用时。用时为整个验证窗口墙钟时间（包括启动/加载），不是纯kernel耗时。

|窗口|北京时间（2026-11-26）|墙钟秒|用途|隔离已解除|
|---|---|---:|---|---|
|r8owner1126a1|21:57:43–22:02:20|277.405|冻结r8 C1、活跃包装竞争、Native配置负例|是|
|r8owner1126a2|22:03:57–22:08:32|275.216|冻结r8 C1及准确ALREADY_INSTALLED分支|是|
|r82owner1126a1|22:09:40–22:13:57|257.265|r8.2候选同探针验证|是|

合计809.887秒窗口时间，单张物理卡占用均为这些窗口；不作为故障性能倍数结论。

远端原件：`/home/17-fault-c2c3-20261117/window-<attempt>/`；蓝区核验副本：本工作树 `.codex-tmp/r82-evidence/<attempt>/`。原件SHA在逐次review中，下载逐文件校验，非凭控制台PASSED结论。

不修改r8标签、wheel或r8.1登记。候选修复、复现代码与审查文档提交开发分支。黄区D/G已使用0–7（v4b-07/19301），A系列使用8–15，互不等待；本轮不操作黄区实例。
