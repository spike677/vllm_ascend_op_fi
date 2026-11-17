# 继续位置

本轮P3/P4的限定实验、本地实现和总结已完成，当前没有运行中的验收容器。最后只读检查卡5空闲。分支为 `codex/worker-p3-p4-20261117`，发布状态和提交号以本分支Git记录为准。

- Stage1 `d1c8a20c`：20 WeightNz / 39 MatMul公共Worker限定正向闭环，真正GLM/DeepSeek正常推理。
- P3 `0c6ec2ed`：15/17/18第二层独立诊断，原严格结构失败不变。
- FIA26 `231e219c`：真实SmolLM/GLM的29/39次writer→consumer→恢复；GLM自比NaN与完整缓存池未恢复仍保留。26 residual新运行分配观察没有复现旧rank1 NaN，来源未闭合。
- 本轮quant：真正DeepSeek-V2-Lite的公共Worker B8首decode，自动选定一个实际embedding/norm/projection。request2的scale及71/2048个Q改变，完整hidden/logits变化，token/text不变，OFF精确恢复。
- 本轮33：真实HF连续4步decode，KV16383→16387，首次合格长度的28层F策略8192→1024，其余为正常策略。B/F各336个实际IFA launch审计通过，完整logits逐步变化而token不变，OFF精确恢复；不是公共Worker资格。

详见README、QUANT_REPORT、FAULT33_REPORT及mechanism_delta。599项CPU测试全部通过，无跳过。

量化两种采集范围都保留：`quant-deepseek-a6`整段prefill＋decode统计遇到写前NaN；`quant-deepseek-decode-a1`完整首decode的step2首差为DynamicQuant output/1 Min 0.13877953588962555→0.14271654188632965，之前163项null，B自比后续仍有NaN。不能称全段严格通过。

33正式attempt为`continuous33-a3`；a1覆盖范围错误、a2采集器包装后OFF源码发现失败均保留。新probe在采集前绑定三个独立session的真实对象，不放宽原发现与覆盖门禁。独立首差四步依次为下游mean/mean/mul/linear，不是IFA唯一定位；所有null保留。运行时证据不参与首差选择。

124根目录：`/home/p3p4-closure-20261117`。冻结包`source-p4-final-a1`，安装`packages-p4-final-a1/installed`；archive SHA为`1929c876f2d5f087215c2e7bb1ce3a619a4f241cc5e3cfe10878ca69c24985d2`。外部probe按case独立冻结，其最终SHA在执行摘要。量化native为`quant-v2-851-a3/quant-v2`，原生V2实际entry计数用于证明隐藏resolver下的命中。

发布只含实现、总结和校验哈希；新原始statistics/mix、完整张量、模型权重与vendor二进制留在124。旧校验清单对应旧提交，当前以`SHA256SUMS_P4`为准。

后续仍未完成：26 residual旧rank1 NaN来源、33公共Worker接线、17条的公共多目标/all_eligible资格、其他部署范围和黄区内源正式接线。不要将本轮限定验收升级为这些结论。

如追加实测，一次只用一张空闲物理卡2或5，使用`/var/tmp/fault22_npu_2_5.lock`和每case空闲门禁、每批至多8case，不停止新外部任务，不恢复过期Docker入口隔断。新attempt保留原件；CPU成功不授予NPU验收。
