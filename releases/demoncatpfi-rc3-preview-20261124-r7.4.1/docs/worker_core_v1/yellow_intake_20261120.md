# 黄区材料接收记录：2026-11-20

> 以下首次接收状态保留。晚间已按负责人最新文本证明决定完成RC2.1登记；最新结论见末节，不再要求传输原始文件。

## 来源与本地保存

负责人转交 openchamber 本轮完整文本汇报，报告声明文档提交为 `e16dfa6`。本次实际收到的是粘贴文本，未收到其列举的 JSON 回执、总结、安装检查或验收 harness 源码；未访问黄区、未核验该文档提交。

原文逐字节保存在仓库外 `D:/DemonCATPFI/intake/yellow/20261120/openchamber_rc2_report_20261120.txt`，接收清单为同目录 `intake_receipt.json`。

- 原文大小：13382 字节。
- 原文 SHA256：`3b6fb59d3599f6c1b6ff8683d2aa21d11f5b4c50d7c4eb10704c094fe1021aaf`。
- 原始文本未进入 Git；本接收记录不转录机器地址、机器编号、逻辑设备编号、容器名或请求/owner标识。
- 本次仅本地落盘，未提交或推送 GitHub，未向 openchamber 派发任务；没有修改冻结标签、发行包、部署登记或准入状态。

## 报告声明的验收范围

以下均为黄区报告所述，尚未通过原始材料独立复核；不作为蓝区验收结果。

|项目|报告内容|
|---|---|
|RC2身份|标签 `demoncatpfi-yellow-rc2-20261119`；源码 `d07c72156e0fe4e3608dcee58cdb422b49834c66`；发行 `de17267bdb7a600b0c97166a009f4a319b65c4fb`|
|组合|A3；Qwen3-0.6B BF16；TP/PP/DP=1/1/1；eager、同步调度、关闭prefix cache与chunked prefill；max_num_seqs=1、max_model_len=2048|
|软件|Python 3.11.14；torch 2.9.0+cpu；torch_npu 2.9.0.post1+gitee7ba04；vLLM/vLLM-Ascend 0.18.0；实际 CANN 8.5.1|
|模型config SHA256|`660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`；报告声明 live_config_matches_file=true|
|安装检查|三个distribution安装到独立目录；IMPORT_AND_DEFAULT_CLOSED_CHECK_ONLY；issues为空；未创建Worker、未加载Native、未推理|
|18.api_half|同Worker两个作用域，normal→fault→off→health；报告F changed_elements=5120，恢复及资源清零通过|
|39.output_proxy|同Worker两个作用域；F changed_elements=1、finite_to_positive_inf=1；恢复及资源清零通过|
|22.wrong_row|同Worker两个作用域；恢复及资源清零通过；本摘要没有F changed_elements或独立效果数值|
|负向检查|无许可创建18被拒且无资源残留；39计划耗尽后再次advance被拒，单独attempt清理完成|
|登记状态|三个总结均声明review_ready=true、registration_written=false、acceptance_granted=false；master仍保持RC1.1|
|其他能力|黄区8/17未验收，Batch待配置；Qwen3.8与DeepSeek V4.1只做只读inspect/结构观察，不替代上述RC2小模型验收|

## 包与总结身份（报告值，非本次文件重算）

|制品|声明SHA256|
|---|---|
|源码包|`1a59219c0ec5e37204f208c12a2a9e6f42edcb491eee82353ccba6a0fae050d8`|
|demoncatpfi 0.3.0 wheel|`57128ac1f310730b7cfd3be803f06f7a3807284b59e500a91a07db8eefe8783b`|
|worker extension 0.2.0 wheel|`94d577b1f95de89825edd4a1e9fc0c59c19248a776e7a441b2b68ba5fa77adb1`|
|fault39 core 0.2.0 wheel|`1e2e85160b983dcec38bde859c8a9ce5fbc649fc707062fde62fdae02701fc1a`|
|source-review.json|`e1622a4bdc91f31576342e9d66e02f75e092f41738bb8fe8b8413fe553aecdea`|
|source-manifest.json|`373fa8a48749e9fdade903535610da11420f439e4ee499af6ddc9c629cd010c2`|
|install-check.json|`9cac20b44e17dd5d57cf7504abfff527ccc99a4d68b8f5e1306e2c6d1fd6f982`|
|39 Native库|`9261d0867f822c90d0d179457e41bc9a2b132bc39e643b61fe5d47a3fd81424f`|
|18 summary|`3316539c924403f09047a67cc71b0e9dd718048b93226dc0f04ee3872b682941`|
|39 summary|`125340081aa5a02dfe5c923393bba86a9ae5f7150e90fcee414b8cd88f24b01e`|
|22 summary|`b099cf352269c97f1d5921849f3a4b1f7f14618c81fc6cf4c2470e799c25f9c5`|
|39负向回执|`5330e1c7212bb2101af59d34314c8d2471952fd5f67777b00bb5d167982a64e1`|

报告中的许可source为`YELLOW_TEST_DEPLOYMENT`；字段`source_manifest_sha256`实际填写上述`source-review.json`哈希，不是`source-manifest.json`哈希。报告称许可逐作用域安装/撤销，不能由用户FaultConfig开启。

## 字段完整性与待核验项

1. 原文列举16项回传文件，本机本次接收目录尚未发现这些原件；逐项缺失清单见下方及仓库外接收清单。文本中的“回传内容包括”不等于文件已到达本机。
2. 未收到三项总结、六个作用域回执及负向回执内容，尚不能复算同Worker身份、真实内部请求映射、实际命中/消费者、完整OFF/health相等及各项资源清零；不从蓝区报告或历史默认值补齐。
3. 22摘要没有效果数值；需查看原始RoPE事件和输出比较后才能判断效果。不能将未上报changed_elements直接解释为未触发，也不能仅凭review_ready认定已触发。
4. 未收到安装校验原件、已安装源码逐文件核对结果、许可安装/撤销回执与RPC包装harness源码。仅包装公开RPC的接线声明保留，尚未完成代码审查。
5. 报告明确未单独保存release-manifest.json、SHA256SUMS、发行README和install-summary.json。本次不声称完成整套发行清单核对。
6. 本次报告将实际CANN更正为8.5.1，区别于上一批文本的8.5.0；保留两批原文。后续登记必须以核实的实际运行值为准，不能为匹配交付单填成8.5.0；本次未登记。报告新增A3及芯片型号说明，原始设备查询尚未收到，唯一设备标识不转录。
7. 新口径把专用实例称为现役服务实例，同时声明未操作对外服务实例；名称、服务用途与验收组合分开保留。不能因名称调整把许可source自动改为YELLOW_DEDICATED_INSTANCE，或把小模型TP1结果扩展到现役大模型/TP2。
8. B8首次decode调度证据、TP2通信采集、DeepSeek严格参数实例、RC3步骤仍未提供；DeepSeek容量两种口径待负责人确认，不作当前RC2小模型登记依据。

缺失文件：

- `install-check.json`
- `rc2_18_scope1.json`
- `rc2_18_scope2.json`
- `rc2_18_summary.json`
- `rc2_39_scope1.json`
- `rc2_39_scope2.json`
- `rc2_39_negative.json`
- `rc2_39_summary.json`
- `rc2_22_scope1.json`
- `rc2_22_scope2.json`
- `rc2_22_summary.json`
- `source-review.json`
- `source-manifest.json`
- `deployment_acceptance_rc2.md`
- `README.md`
- `rc2_status.json`

## 用途和后续归属

- **组合登记**：作为RC2黄区审查的待核验摘要；原件未齐，暂不足以发布RC2.1或授予acceptance/product_closed。
- **现役摸底**：仅更新报告所述状态，不把摘要中的Qwen3.8/DeepSeek信息当作已收到完整inspect证据。
- **缺陷定性**：CANN版本口径差异及pcp_size部署不匹配沿现有边界记录；本次不据摘要认定新的RC缺陷。
- **codex**：收到原件后复核包/安装/许可/harness/双作用域与负向检查；再决定登记和维护发行。
- **openchamber材料缺项**：上述原件、接线源码与实际运行身份；由负责人按既有交接安排转交，本次不直接要求黄区执行任何操作。
- **负责人**：实例术语与许可登记类别的对应、DeepSeek容量口径及后续阶段安排。

## 2026-11-20 晚协议更新（取代上文的原件转交要求）

负责人已明确黄区只能回传纯文本。上文“未收到原件”仅保留为当时接收事实，不再作为要求转出原始文件的条件。新的登记证据等级为 YELLOW_TEXT_ATTESTED，原件留黄区；来源类别仍为 YELLOW_TEST_DEPLOYMENT，不与证据等级混用。

当前叙述性报告仍缺：三份 summary JSON 全文、六份原始回执文件名/字节数/双哈希与关键字段提取文本、install-check 全文、负向清理状态、CANN 运行读取来源、RPC 包装源码全文。具体只读提取方式见 [yellow_text_return_kit.md](yellow_text_return_kit.md)。文本齐全且相互一致后即可审查登记，不要求原件传出，不用蓝区历史补齐。

RC2 summary 的 source_receipt_sha256 是规范化 JSON 哈希，不是原文件字节哈希；补充工具保留两个值并绑定 owner。当前不创建 RC2.1 分支，不修改登记表，不授予 product_closed。RPC 正式可信部署 binding 入口列入 RC3 接口事项。


## 晚间追加：文本证明登记已发布

已审查负责人指定仓库外 D:/CodexArchives/20261120/yellow-rc2-user-results。18、39、22按明确决定登记YELLOW_TEST_DEPLOYMENT，证据YELLOW_TEXT_ATTESTED，原件留黄区。组合A3/Ascend910_9382/CANN运行8.5.1（toolkit文件8.5.0）/vLLM-Ascend0.18/Qwen3-0.6B BF16/TP1，不等于现役模型。

18两作用域非同Worker，39/22同Worker，保留实际限制。22行级效果未进入旧summary（工具缺口），以hidden/logits变化和OFF/health精确恢复的文本证明登记；RC2.2起summary输出Q/K或rotary行变化，不补造旧材料。RPC子类仅调用公开binding安装/移除、未放宽准入，接受。完整install-check只有文本摘录，披露证据限制。

RC2.1源码8c358961d9db16bd2bcbf6916bd8d9f5722d3a18，发行7576976ec63ff3d440ebe9c8090c003a022f1203，标签demoncatpfi-yellow-rc2.1-20261120。机制源码只改登记表；升级步骤已随新发行提供，待负责人转发，没有代为向黄区发任务或升级master。
