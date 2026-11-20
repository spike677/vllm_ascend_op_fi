# 2026-11-20晚执行与交付

截至10:16 UTC。冻结旧标签和包未修改；没有向黄区发送内容。

## 开发提交同步

开发线bd1990e7、0fbcc442、e4268a53加入Batch/41 TP2、公共owner与Native owner绑定及rank守卫。41/恢复/多rank CPU集合41项通过；Batch普通入口18项通过。后者修复测试间模块缓存污染后重跑，不将CPU通过当作NPU验收。新统计原件、模型和完整张量不提交。

## RC1.x / RC2.x维护与发行包

|版本|源码提交|发行提交|核验|
|---|---|---|---|
|RC2.1|8c358961d9db16bd2bcbf6916bd8d9f5722d3a18|7576976ec63ff3d440ebe9c8090c003a022f1203|124 aarch64无设备安装，285个Python文件逐字节一致|
|RC2.2|ec941ed6d2a166443d208e1bb9a3284667ae4cb1|0e4b2e4a356cef05958c02cae7c4b924dd450e4e|842固定Git文件导出，290安装Python文件一致；137 CPU项通过、1跳过|

[RC2.1包及UPGRADE.md](https://github.com/spike677/vllm_ascend_op_fi/tree/demoncatpfi-yellow-rc2.1-20261120/releases/demoncatpfi-yellow-rc2.1-20261120)；[RC2.2交付](https://github.com/spike677/vllm_ascend_op_fi/tree/demoncatpfi-yellow-rc2.2-20261120/releases/demoncatpfi-yellow-rc2.2-20261120)。两远端标签及分支已核对，安装NPU用时0。RC2.1保留旧包版本以遵守只改登记表要求，须用新空target、精确wheel SHA、no-cache安装。RC2.2三包版本0.3.2/0.2.2/0.2.2；包含41/request_plan/28研究标注/缺项汇总/TP1第二作用域及22效果总结，不含多卡代码。测试部署线冻结后仅维护。

黄区登记A3/Ascend910_9382/CANN运行8.5.1、toolkit8.5.0/0.18/Qwen3-0.6B BF16/TP1，18/39/22为YELLOW_TEXT_ATTESTED。18两作用域非同Worker；22旧行级效果缺项按负责人接受的hidden/logits和恢复文本证明披露，RC2.2修正工具但不补造旧值。升级步骤交负责人，不直接发送黄区。

## 逐项闭环 / 多卡主线

41 TP2完成：A2/CANN8.5.1/vLLM-Ascend0.18/Qwen3-0.6B BF16/TP2，同一组Worker两个owner，共12实际请求。自动选中第一实际Attention的value-cache尾列；rank0每次Native写8元素，rank1边界不写；两rank完整decode hidden/logits变化，完整prefill/decode OFF/health哈希精确恢复。显式recover后finish的hooks/leases/inflight/replay_pending为0并detach。占卡10:12:58.888321–10:16:02.468283 UTC，183.579962秒，含初始化和清理。

[独立审查报告](../../experiments/worker_core_v11_20261118/m1_kv41_tp2_20261120_summary.json)重新比较原始阶段、Native effect、完整哈希和finish回执，不读验收驱动checks断言。guard改值、OFF哈希变化、第二组Worker、残留lease、漏命中五个内存负向案例均拒绝，原件不改。仅合法KV地址代理，不声称实际越界，不唯一归因某个collective；BLUE组合不授A3/现役资格，不进入RC2.x。

Batch TP2未通过：A2/CANN8.5.1/0.18/Qwen3-0.6B W8A8/TP2/B8。a1租约身份问题已修复；a2/a3两次normal均拒绝BOUNDED_COMPLETE_NATIVE_CALL_REQUIRED，两rank Native before_sequence=after_sequence=0、prepares/executes=0。无F及OFF通过声明。按同因两次停止，不计为设备冲突。a1/a2整窗197.382900/173.605057秒，a3 Batch占卡166.553805秒。TP1旧通过不扩展TP2。18/22/39 TP2沿用既有同Worker两作用域和显式recover通过，未重跑。20/26两变体/33维持止损。

## 28限时进度

B2依负责人决定接受stock/private完整哈希桥接，保留stock census FUNCTION_BIND_NOT_UNIQUE及health仅token限制；M128控制已止损。B3–B5仍未通过，Native两阶段F准入尚未接通，不能用typed旧JSON/日志替代Execute当前prepared输入。只读核对实际生成AddInput/AddDynamicInput、GWS/Execute和SDK接口，未启动新28 NPU测试。剩余13730.811623秒，两工作日从11-20 02:26:37 UTC起，不延长；公共入口关闭。

## 现役路径 / M2门禁

官方[A2教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-27B.html)有qwen3.8-a2镜像记录，实际pip栈未安装核对，不能凭tag称为完整0.26目标。ARM64 manifest sha256:910812c158b44c41eb30f896ab780b40bf4179431eb0d34581adba84f8d36575，15层压缩6705149643字节。

流式测量09:18:02–09:40:13 UTC，不落盘层数据。前三层校验通过、解压共1113710080字节；第四层预期压缩4280972307，只收到945815552、临时解压2909798400字节后gzip EOF。按门禁停止，总解压大小未证明，不docker pull、不改Docker/驱动/CANN/0.18。存储释放不能替代完整预算。

社区W8A8的25文件与固定公开快照SHA一致；config SHA191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab同黄区，但quant项数W8A8_DYNAMIC1843/FLOAT585，黄区1321/759，不是同一量化制品。crc32.txt首个chat_template预期cf59f3ab、实际367cbad5，而文件SHA符合公开快照；保留公开清单内部不一致，不改文件、不声明CRC全通过。M2安装、重构建、Qwen3.8 TP2尚未执行。

## 存储整理

旧模型及获批别名重新核查无其他引用后删除，无归档。09:11:06 UTC，du42773396798字节；/home可用58130685952→100904173568，增加42773487616字节。指定CATMonitor37标签/20镜像ID无任何容器引用后逐项rmi，09:12:18–09:12:46，/home100904005632→102641094656，实际增加1737089024字节。无prune -a、无其他项目删除。

完整前后df/du、路径/字节/mtime、引用检查、删除记录：仓库外D:/DemonCATPFI/server-backups/124-storage-20261120。模型删除回执SHA f90ec273f3193494fe0a2cd3c5925bf08be035cc01fe97fcf395dfed96b6b2ee；镜像删除日志SHA45d501c3595af1725760a6f0506187307d26ddaba2213b48ab457121ef2fe7c4。c2c3增量备份/分类尚未执行，依赖保留；/home/haoran/experiments未删除。

## 抢占与隔离记录

只停止明确映射到{2,5}且不涉及其他卡的非项目容器；共享其他卡则跳过。资源锁+最长1800秒清理线程，结束join解除；没有修改派发器配置、Docker入口/daemon、驱动/CANN，没有删除或重启外部容器。

|窗口|UTC开始及结束|停止对象|恢复|
|---|---|---|---|
|a1|09:51:47起、约197.383秒|卡2 Expm1、卡5 Abs|清理器结束，隔离解除|
|a2|09:58:12起、约173.605秒|卡5 groupedmatmul|清理器结束，隔离解除|
|a3|10:10:06.915088–10:16:03.500061|卡5 groupedmatmuladd|cleaner_stopped=true，未改派发配置|

完整容器名、镜像、停止时刻、恢复动作在各controller/isolation.jsonl及本地私有验收目录，不公开扩散外部任务标识。最后窗口停止镜像cann:9.1.0-910b-torch2.10.0_npu-atk26.7.8的卡5实例。本轮未见设备冲突；Batch拒绝是调用链缺口。yellow_text_return_kit仅保留已写部分，停止扩展；不再写卡窗口申请。

恢复复核：10:16:19.733851142 UTC，外部调度器已在卡5重新启动同类resume实例（绝对时刻来自Docker状态）。本项目只读确认，没有再停止它；说明窗口清理器解除后实际派发恢复。

提交前共享TP1/TP2 CPU回归63项通过；测试集合有重叠，不与先前41/18累计。
