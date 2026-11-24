# r6产品模式迁移（2026-11-23）

P0独立提取工具已交付于46187caa；r5未改，r5.1等待黄区提取证明，不等待summary全文。r6尚未冻结。

## Batch：A2实测通过

组合：Ascend910B4 A2、vLLM0.26/vllm-ascend0.26rc1、CANN9.1、Qwen3.8社区W8A8、TP2。max_num_seqs=8，max_num_batched_tokens=8192，max_model_len=1024，纯文本、单chunk、无prefix缓存、无异步调度，每请求生成4token。不是A3产品闭环。

真实HTTP /v1/completions并发用户请求；宿主仅attach/enable/status/disable/detach，Worker从实际scheduler映射获得ID与批次。每作用域提交2/4/8并发组、B1负例和5token范围负例，共16条用户请求；同组Worker两个独立owner。实际出现decode B2/B3/B4/B7/B8，不能将HTTP并发数当成调度batch大小。

仅全decode、2～8条受支持请求的当前调用全真实行共用absmax/127；原QMM消费原位更新Q/scale。预填充、混合批、B1、超出范围请求不注入并列原因。共享调用在每条请求中只表示参与；独立审查器按rank和frame去重，核对所有参与者与跨rank映射。

每rank每作用域：observed=7296、eligible=attempted=applied=consumed=restored=2432、failed=0。changed分别为作用域0两rank2400/2400，作用域1为2400/2399。产生变化与成功应用分开记录。按原标准核对scale精确、抽样Q±1、原QMM消费者、区域hash恢复；关闭后金丝雀两次连续有限且完全一致，关闭前后hidden/logits精确，Runtime owner/hooks/leases/inflight清零。公共自动入口source_manifest SHA为2c438c2e472b6e1699f6f70f2d2fa84363fb0c68b1845b86ceb44750c1633729。

完整回执保留124，提交[独立总结](review_batch.json)，未上传完整私有数组。审查器不读取注入日志选择首差；本轮产品验收也不额外要求msprobe首差或token改变。

窗口02:51:54.536163～02:57:23.850370 UTC，329.314秒含模型加载/清理，不当作纯NPU时间或性能倍数。只停止卡2/5已授权外部容器；所有停止名称/镜像/绝对时间及恢复见总结isolation。结束已停止有界清理器并恢复派发能力，无Docker入口/驱动/CANN修改，无数据删除。vLLM退出时EngineDeadError与共享资源警告另行保留，不作为故障区域恢复证据；容器已结束。

## 待完成

39mr和33公共HTTP产品模式已通过独立复核。33仅覆盖方向A的A2实测范围2304-token prompt/2305～2307 decode，不把>128泛化成全部长度。A3默认拒绝复用A2分组，先做实际launch→控制重算≥99%→两作用域。Native三项等待可信A3构建审查再迁移。DeepSeek等待噪声/只观测提取文本。


## 39mr：A2实测通过

同一新栈/模型/TP2，max_model_len/max_num_batched_tokens/max_num_seqs=1024/1024/1。同组Worker两owner，每作用域3条不同用户prompt和一个5token范围负例；宿主无逐请求钩子。每有效请求每rank observed1216，eligible/attempted/applied/consumed/restored704，failed0；changed为701～703，未把舍入后无新变化误记失败。累计每作用域每rank2112次符合条件调用。所有实际调用均核对CPU FP64完整K控制后才执行omit_final_quarter，非目标区域一致、真实消费者与恢复hash齐全。非量化和真实QMM投影分别适配；512次/请求不适用原因保留。

关闭后新金丝雀hidden/logits精确恢复；所有故障租约/钩子/在途/owner清零。原始完整用户输出不作逐请求normal对照，本验收依据本地同输入控制+部署金丝雀，不宣称用户每个答案都改变。

[独立复核](review_39mr.json)，实际source_manifest=437c91c25b577b4299b1fe9653123bed7622fee608aac072500ae4e41e14b2e5。03:05:37.715464～03:10:23.210807 UTC，285.495秒含加载/收尾，停止外部容器0个，隔离已恢复。第三方进程退出shared_memory/semaphore警告保留，不能将故障资源清零推成整个vLLM无退出告警。

## 33：A2限定范围实测通过

同一组合，max_model_len/max_num_batched_tokens/max_num_seqs=4096/4096/1；3条不同用户prompt各2304 tokens、生成4 tokens，另有5token范围负例。同组Worker两个owner。真实decode长度2305～2307，每有效请求每rank observed64、eligible/attempted/applied/changed/consumed/restored均48、failed0；每作用域每rank144次符合条件调用。16次prefill不注入。消费者为AscendRowParallelLinear.forward；每个符合条件的完整目标行替换前均核对控制重算逐位一致率≥99%、同一合法key集合、不重复分母、无逐值挑选。不能将最后一个检查点的100%当作所有调用100%。关闭后全rank区域字节恢复、故障资源清零、新金丝雀hidden/logits精确恢复。

[独立复核](review_33.json)，实际source_manifest=5cff97a3467101b5b214efdaba9404cd7d8281130d3c9dac376e555b5b88ef11。03:11:44.555863～03:16:29.013546 UTC，284.458秒含加载/收尾；隔离已恢复。退出时第三方semaphore/shared_memory警告仍保留。审查工具修复了33控制检查局部变量覆盖八项计数字段的问题，并重审同一原件；原件未改，新增回归测试验证计数保留及98%控制被拒绝。

## CPU与发行边界

完整候选CPU集运行1096项，0失败、0错误、2项既有研究路线skip（详见[cpu_review.json](cpu_review.json)）；后续针对Batch兼容性、产品准入及独立审查的45项测试全部通过。P0提取工具的50项检查另行记录在独立补充件。未改r5冻结物、未登记新黄区组合、r6尚未冻结。实际运行包的source_manifest与随后只改审查器/文档的开发提交分别记录，不将后者冒充实机运行源码。
