# DeepSeek-V2-Lite 产品预演：BLUE_COMBINATION

2026-11-23，124 A2 / Ascend910B4，vLLM0.26.0、vllm-ascend0.26.0rc1、torch2.10.0+cpu、torch_npu2.10.0.post4、CANN9.1.0、Python3.12.13；DeepSeek-V2-Lite-W8A8、TP2、EP关闭、eager，max_model_len/max_num_batched_tokens=1024。Batch max_num_seqs=8，其余为1。实际身份及模型三类SHA见 `summary.json`，不以Qwen或V4元数据代替。

六项均完成ProductClient + 真实HTTP用户请求、Worker自动发现、两个owner；每项两个owner使用相同Worker组。故障请求含prefill和后续decode，共4个生成token；每个rank每次符合条件的调用持续触发。未要求token必须变化。结果是**限定范围的蓝区产品预演**，不代表V4、MC2、EP、A3或黄区主项目闭环。产品免许可登记仍未打开。

|机制|每owner、每rank observed / eligible / applied|changed（owner0 / owner1）|范围|
|---|---|---|---|
|8.allgather_output_row|312 / 78 / 78|78 / 78|真实AllGather MoE unpermute输出行错位；decode单行234次不适用，不制造调用|
|18.api_half|660 / 660 / 660|660 / 660|真实DeepSeek风格RMSNorm输出及其消费者|
|39.output_proxy|1389 / 996 / 996|996 / 996|BF16 ND投影返回；393次非BF16 ND不适用|
|39.matmul_range|1389 / 336 / 336|336 / 336|已支持的本地列分片，完整K对照；其他分片/格式拒绝|
|quant.batch_shared_scale|2073 / 664 / 664|651 / 654|独立DynamicQuant→原QMM；按rank/frame去重，不累加每个请求的参与计数|
|28.fia.denominator_replay|324 / 324 / 324|324 / 324|27层MLA；expanded prefill与absorbed latent decode，原生copy→o_proj|

两rank计数相同。所有eligible均attempted=applied=consumed=restored，failed=0；changed小于applied可表示结果原本相等，不算失败。各owner/rank八计数、元素数、消费者、恢复和原回执SHA在独立 `review_*.json` / `summary.json` 中。原始回执保留在蓝区私有运行目录，不提交完整张量。

## 真实产品调用与负例

每个普通owner提交3条不同用户prompt及1条超范围请求。Batch每owner提交并发2/4/8组、单请求、超范围请求，共16条，实际合批大小包括2/3/4/7/8。调度不保证提交并发数等于真实decode批大小，回执按实际frame判断。B1不注入并记录DECODE_BATCH_SIZE_ONE；prefill/混合批次不注入。

各项均核对5-token请求不注入、未知profile、未知mechanism、用户指定Native路径拒绝，以及清理后的无binding/无许可拒绝和资源清零。后者同时缺binding/实际调用，**不是仅删除许可的单变量负例**，不据此宣称许可隔离已被独立证明；精确拒绝码保留在summary。开启后的合法请求提供对应产品路径正例。

各owner关闭后：注入区域字节精确恢复、租约/owner/hook/在途资源清零、同一Worker干净金丝雀的hidden/logits完整哈希精确恢复。Bootstrap是r8开发中的可信入口；这些仍是带验收许可的产品模式，不是免许可生产登记。

## 本轮发现并修复的共享路径问题

1. **Batch静态候选不等于真实调用。** 首次在不注入预热阶段触发PRODUCT_BATCH_SITE_FORWARD_COVERAGE。旧逻辑要求每次前向调用所有模块；MLA吸收和MoE融合使部分对象未走其apply。现为每次真实scheme.apply入口计数与事件逐项多重集对齐，少记/多记均拒绝；未调用的候选独立列出。未改模型/量化内核。
2. **Batch范围限制必须保留。** 每owner跨两rank合计1248次 `npu_dynamic_quant` 在已发现独立linear scheme外，原样透传并报告 `OUTSIDE_DISCOVERED_LINEAR_SCHEME`，不计eligible。融合专家/GroupedMatmul量化未适配，本结论不声称全部模型量化都被覆盖。存在但本前向未调用的候选名称亦完整保留。
3. **28实际消费者经过真实复制。** 第一次运行缺consumer_operator，独立审查拒绝；第二次加强存储检查后发现返回结果与o_proj输入不是同一存储。精确审核的mla_v1.py SHA为 `67b4479cca16fd164b33b91b2fa4f48e95f2474720e484abdf3b3897a8ea1545`，其forward确实执行原有的 `o_proj_input[...] = output_prefill/output_decode`。新MLACopyMode只观察这次原生aten::copy_、源对象区间、完整复制值；o_proj读的是其目的区域，设备同stream事件完成后才记consumed并恢复注入的临时返回区域。没有人为补copy_，没有仅以两个张量相等推断关联。
4. **8的新旧机制分开。** `8.allgather_output_row` 是新产品命名变体，保留原路由metadata和真实unpermute后改输出；不把它冒充 `8.restore_offset` 索引故障。A3/V4/EP/TP1和旧Tensor研究factory不能借用此入口。

失败回执及SHA、失败窗口均保留；后续成功不覆盖失败原件。28分母重放是大误差效果代理，不声称历史kernel复刻。39mr仅限定算术代理；其余未覆盖的模块不因共享框架而自动获得资格。

## 窗口与恢复

|窗口|秒|结论|
|---|---:|---|
|v2-product-8-a1|196.064|通过|
|v2-product-text-a1|250.165|18/39op/39mr通过；28旧证据拒绝|
|v2-product-batch-a1|185.181|预热覆盖规则拒绝|
|v2-product-28-a2|171.493|错误的直连存储假设被拒绝|
|v2-product-batch-a2|254.665|修正后通过|
|v2-product-28-a3|192.652|真实复制链修正后通过|

合计1250.219秒TP2窗口墙钟时间（含加载、预热、验收、失败和清理），不是kernel时间或单请求性能。用户请求耗时见summary；没有同prompt正常/故障成对计时，性能倍数保留null，不作门禁或性能结论。

每窗口只使用卡2/5；原有有界exact-device docker stop清理器在窗口结束后退出。停止外部容器名称、镜像、绝对时间及恢复写在 `isolation_*.jsonl`。3次仅解除经脚本SHA和卡范围核对的外部pair租约，见 `pair_preemption_*.json`，未改派发配置、heartbeat、Docker入口、驱动或CANN，没有删除外部容器/数据。

最终只读核查 `window_exit_check.json`：本轮项目容器为零，pair锁已由外部租约重新取得，证明派发能力恢复。解释器退出时仍有第三方resource_tracker的semaphore/shared_memory告警，不能把它等同于已证明这些OS资源泄漏；项目租约/owner和容器退出检查分别记录。

## 复核入口

`tools/review_product_mode.py` 对私有原回执执行独立审查，explicit hardware_scope指定本组合；`review_*.json`记录审查器SHA。`summary.json`另核对两rank真实身份、owner、八计数、负例和恢复。实现入口：continuous_binding、product_batch、sites/allgather_output8、sites/mla28；V4接线准备见上级 `r8_dsv4_and_paged33_preparation_20261123.md`。

仅同步r8开发分支，不冻结、不下发r8，r7.1及以前发行包和标签保持原样。原件映射及SHA足以定位蓝区私有回执；本总结不能代替黄区独立产品验收。
