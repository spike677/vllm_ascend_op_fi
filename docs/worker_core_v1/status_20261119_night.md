# 2026-11-19 夜间只读现状审视

快照基线 bca543467556d061bd46cdb6884a5c9886c32f0b。此审视没有启动 NPU；不向黄区发送内容。冻结物不改。黄区 master 保持 RC1.1，RC2.1 只登记经审查的黄区组合，当前不开维护分支；41及新增能力另走 RC2.2。

## Git 与工作树

以实时 `git ls-remote` 核对，并非只依赖缓存 tracking ref。

|线/标签|本地发行/开发提交 = 远端提交|源码提交|状态|
|---|---|---|---|
|codex/product-integration-20261119|bca543467556d061bd46cdb6884a5c9886c32f0b|同左|干净、无未推送|
|codex/yellow-rc1-20261119 / demoncatpfi-yellow-rc1-20261119|efb849e90f25bd1e0f25199d2865ac542f31dfab|a315d89c962c93eacd61546bb9188a3fdbe424a2|标签对象 b7680dc66c8537e01ae6874aea3c4b190b349b68，本地远端相同|
|codex/yellow-rc1-1-20261119 / demoncatpfi-yellow-rc1-1-20261119|70db3e09465f1ca98f4a4cd0436a4096efd4b592|e39d48c1a86f629241e4edc3817048544f1bf722|标签对象 cfb4c9f623179146bafbb4844ce615c9ce449c7e，相同|
|demoncatpfi-yellow-rc2-20261119|de17267bdb7a600b0c97166a009f4a319b65c4fb|d07c72156e0fe4e3608dcee58cdb422b49834c66|标签对象 867ae4da91fb26c5a7df86c456bd57c963c05907，相同；无独立 RC2 维护分支|

当前主工作目录 `D:/DemonCATPFI/worktrees/fault28-native-fa-publish/demoncatpfi/.publish/worker-core-v11-20261118`；共享 Git 位于 `D:/DemonCATPFI/repository`，C盘登记路径通过既有迁移入口使用，未再搬移。
两个储备树均在同一 `.publish/` 下，HEAD均03ad3b397876fb631e54b3850b9a134fa8795038、工作区干净：`native-ordinary-rc2-20261119`预留20/26/33/41；`rc2-tensor-ordinary-20261119`预留8/17/22。实际开发集中在主树，不在储备树重复实现。
旧 `codex/worker-core-v11-20261118` 本地 b11d77db，远端0e3df66c，仍有3个未直接推送提交（3dbcdd24、a194e399、b11d77db）；不因此推送旧支线历史。TND储备工作树仍有7项修改、2项未跟踪文件，原件保留待审。故“开发线已同步”不等于所有历史工作树无改动。

## 能力逐项

下表“发行”指普通入口可执行子集，历史源码随包附带不等于获准执行。S = 124 Ascend910B4/A2 32GiB、CANN8.5.1、torch2.9.0+cpu/torch_npu2.9.0.post1+gitee7ba04、vLLM/vLLM-Ascend0.18.0、TP1/PP1/DP1串行eager；具体模型/dtype另列。作用域数只统计普通入口，不把历史C2/C3调用数换算成任务数。N/F/O/H = normal/fault/off/health。

|能力|普通入口状态|蓝区组合、作用域数|验收源码；清单SHA/证据键|发行|黄区|请求阶段|未完成原因|
|---|---|---|---|---|---|---|---|
|8 restore_offset|通过单目标|S，DeepSeek-V2-Lite W8A8，2，同Worker|16f40f1f；E8|RC2|蓝区限定|N/F/O/H|非现役V4.1；不再做DeepSeek实测；黄区部署未验|
|15 alias_detach|当前部署不适用|无普通验收；历史构造NZ机制不授模型资格，0|null；无普通验收清单|无|关闭|无|缺自然私有视图写入→原base消费链|
|17 spatial_padding|通过单目标|S，Qwen3-VL-2B BF16，2，同Worker|16f40f1f；E8|RC2|蓝区限定|N/F/O/H|Torch Conv3d代理，非Native ABI/全部视觉层|
|18 api_half|通过单目标|S，Qwen3-0.6B BF16，Source126单作用域1|ce597d2d；E18|RC1/RC1.1/RC2|用户报告RC1.1两作用域；RC2未登记|N/F/O/H|黄区新部署/许可回执待审；蓝区旧单次不冒充两作用域|
|20 buffer_substitute|普通适配待做|普通0；历史DeepSeek QMM修复Source46，模型身份有null|普通null；历史agent7_native20_independent_review.json，不能代替当前清单|无|关闭|普通待定义；保留discovery1/2、R/L/qualification_OFF/F/OFF/disabled_OFF|旧report会collect、research回落；当前Qwen W8A8尚未资格核验|
|22 wrong_row|通过单目标|S，Qwen3-0.6B BF16，2，同Worker|4c56701f；E22|RC2|待独立RC2部署验收|N/F/O/H|黄区未登记；非整数溢出/全层|
|25 reduce_scatter|不支持|普通0；历史8.5.1/9.1校准均无样本|null；无普通清单|无|关闭|无|NO_REPLAY_SAMPLE；不制造输出替代|
|26 residual_stomp|普通适配待做|普通0；历史Source54/68自然QMM单点，非完整135目标|普通null；Source54 ce82ca5f历史；清单未在本次普通证据中核定|无|关闭|普通待接；保留实际inventory/R/L/qualification_OFF/F/O/H|必须真实独立块邻接；135中27项未解决，不伪造相邻|
|26 fia_destination|普通适配待做|普通0；历史Source62/89 SmolLM自然推理TP1，非训练|普通null；Source62 dc40a87d，E26|无|关闭|普通待接；历史资格与F/OFF保留|typed FIA及allocator当前邻接要重验；历史模型非当前Qwen|
|28 legal（FA/FIA）|历史保留，关闭|普通0；旧限定模型/形状不继承至Worker|普通null；历史证据保留|无|关闭|无当前普通协议|本轮不重开legal路线|
|28 denominator|TND正常链已观察，F未接通|普通0；S、Qwen真实TND，仅正常关联|普通null；Source123正常关联不是F清单|无|关闭|正常观察；F/OFF未验|Execute借用/typed plan待审；不可继承旧HF结果|
|33 Source105|研究单点通过，普通待接|S但FP16，Qwen3-0.6B，首decode Q1/KV1025/page16；普通0|8796f84d；E33|无|关闭|R/L/qualification_OFF/F/recovery_OFF；新增独立health待接|新包须重新stock/private正常桥接；27个匹配点未授proof|
|39 output_proxy|通过单目标|S，Qwen3-0.6B BF16，Source116普通1|134e1188；E39|RC1/RC1.1/RC2|用户报告RC1.1两作用域；RC2未登记|N/F/O/H|受控有限值→Inf代理；不等于历史越界，黄区回执待审|
|39 matmul_range|历史Native保留，非普通子集|普通0；历史Source42修复路线，非全8.x/9.x|null普通；历史构建范围见CONVERGENCE_DECISIONS|无|关闭|历史校准/F/OFF，非普通协议|未适配普通任务；不能用output_proxy替代|
|41 tail_column|开发线单目标通过|S，Qwen3-0.6B BF16，2，同Worker|105fb2c1；E41|仅开发线，拟RC2.2|关闭|N/normal_repeat/L/F/O/H|未发行；hidden改变，token未变，logits未采，不授tiling证明|
|Batch shared_scale|普通6批通过|S，Qwen3-0.6B动态W8A8/B8，a1/a3共2个独立Worker作用域|32f87374+f83d25cb；EB|RC1/RC1.1/RC2|待配置，未登记|前health→N/R/L/calibration_OFF/F/O→后health|两个owner不在同Worker；不能声称满足新增同Worker要求；完整路由泛化未验|

清单身份（SHA256，均为对应实测source manifest，不是RC2许可用source-review）：

- E8 `bcc6759f1a1c9f8a25a825a6379d2e4b0193aadcb69fc7fc76627c0af30956fe`；[8/17总结](../../experiments/worker_core_v11_20261118/ordinary8_17_a5_20261119_summary.json)。
- E18 `c36875d564665c9163fd223bb9fc5eb1089b1b1dd4914a19dd713d6e110cec3b`；[独审](review_rms18_ordinary_source126.md)。
- E22 `15a6eb3895e59e27c8eeef5680eea8df73c5a53cf3cf0207000dcba6a8034f94`；[总结](../../experiments/worker_core_v11_20261118/ordinary22_a1_20261119_summary.json)。
- E26 历史Source62 `352bb926eb5f0ea73612413aae743546e05724889f37b9d03617cdc10bf01a67`；[独审](../../experiments/c2c3_closure_20261117/fia26_source62_review.json)。
- E33 `4a4bb617af27ecaca8232721782f8a1f3974a6e4fcb8081cb4c6e7c8018c2eb6`；[总结](../../experiments/worker_core_v11_20261118/attention33_source105_single_summary.json)。
- E39 `aa40d9908ebae4b5063f32dadf6fc998a2afcb4b5db4afc182ad4ff6633d15e4`；[独审](output39_source116_single_task_review.md)。
- E41 `2e723e2cb8513bd01e690cb9f2972dbf4015c8986476c75b3dd7a89f2e86f87d`；[总结](../../experiments/worker_core_v11_20261118/ordinary41_a3_20261119_summary.json)。
- EB a1 `d0c02dde6002ca45e4401e8a2961a757267d5f4339dd77f1e0e5efa34e300096`，a3 `199b0d9a2154179c0f67839c92a3ff22ff5cfea04576a532afd32c1510724959`；[a3总结](../../experiments/worker_core_v11_20261118/ordinarybatch_a3_20261119_summary.json)。

## 0.26与未关闭事项

原门禁记录：DockerRootDir=/home/docker-system，非根分区，约61.64GiB空闲；17压缩层6718398624字节。精确arm64 digest及CANN9.1.0配置见 v026_prepull_gate_20261119.json。解压量和25.5.0.b060/7.8.0.5.201精确官方配套未证明，停止拉取。最新批准仅先做流式内存计量及官方依据检索，结果另存，不覆盖原记录。

|问题|归属|下一步|
|---|---|---|
|RC2许可文件SHA、binding、next_phase路径不统一|codex|补充单与CPU符号核对；接口统一留RC2.2，不改包|
|上一轮黄区回传未到|openchamber|待回传；本轮不发送新任务|
|RC2.1人工登记|codex|收到完整RC2两作用域证据后审查；当前不建分支|
|何时转交补充单/部署窗口|负责人|待上一轮回传与补充单齐备后决定|
|0.26解压大小及官方驱动依据|codex|批准的只读流式/文档门禁；不通过不拉取|
|20/26两变体/33普通化|codex|按预判逐项；同因两失败停止，报告不适用或实现缺口|
|28真实TND F/OFF|codex|审查既有未提交改动；完成前关闭|
|41发行及新增接口修正|codex|另冻结RC2.2，不能进入登记专用RC2.1|
|A3现役摸底/精确模型制品|openchamber、负责人|Qwen→DeepSeek V4.1 W4A8→W8A8；无GLM，当前无DeepSeek实测|
