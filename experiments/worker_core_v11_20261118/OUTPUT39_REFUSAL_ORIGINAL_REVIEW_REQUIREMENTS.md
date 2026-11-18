# Fault39 一次拒绝诊断：完整原件独立审查入口

本清单用于 source101（commit `417d43ccc85b58554d243791546f9d484fdcd8f1`，source manifest SHA256 `6c6b544f1709f6235961858d175bcffd6494ba47b7bbc556cf50cca49ebd715c`）的在途诊断。它不修改正在运行的代码，不操作124，不将 CPU 构建、窄5条观测或预期 exit2 当成 C3/E5。

root 已报告生产 CPU 25 tests/0 skip，其中实际 `postwrite.cpp` 的 ACL stub 编译运行通过；新 SDK 构建为 `CPU_BUILD_COMPLETE`。这些是父任务提供的当前状态，独立实机审核仍以完成后的原件为准。本文件不提前声称 NPU 结果。

## 最小原件集合

请在本次 attempt 完成后一次性冻结；原JSON字节、路径布局和文件SHA保持不变，不制作改变字段的“兼容旧 schema”文件。可以压缩重复 metadata；不需要任何 `.bin`、`.pt`、权重、完整激活或图像。

| 原件 | 必须保留的作用 |
|---|---|
| `probe/result.json` 与外部提供的SHA | 完整请求顺序、真实 enqueue 映射、失败、abort、恢复请求、原件引用、模型/输入/Native来源、普通健康请求及最终清理 |
| `probe/host_receipt.json` | 原公共 attach→owner/generation→advance/query/collect→recover/unregister/detach 的完整 history；不能只保留最后 `ownership_retained=false` |
| `probe/worker_core_rpc/*.json`，完整有序集合 | 每一个真实 RPC 的原参数与原响应/异常，包括 setup、原Core动作、install/finish diagnostic、remove binding、路由检查；文件数和序号必须与 result 引用完全一致 |
| 被所有 `requests/recovery_requests.original_observations` 以及 RPC `group_originals` 引用的 `output39-runtime/.../group-originals/*.json` | 完整的逐次不可变前缀、C2全量组、失败C3及partial inventory、所有 Native lease/event、独立发现、真实输出摘要和恢复，而非只交最后一个摘要 |
| `probe/refusal-diagnostic.json` | 同 lease.original 的位模式计数/首索引/完整SHA，与两个 fresh control 的五条窄观测；需与finish helper响应中的SHA、大小/路径关联 |
| 本attempt controller/queue manifest、原stdout/stderr | 实际source/image/card/mount/argv与退出状态、软/硬截止、外部中断、设备门禁及清理；若失败于超时/外部终止必须保留，不能改名为guard复现 |
| source101原manifest和实际入口/依赖源码 | 核对公开Core、Native wrapper、observer、helper、原driver及所有loaded source pins；新文档不属于已运行source101来源 |
| 新Native SDK构建manifest/完成回执/构建指纹及原编译命令、编译器/SDK/image身份 | 将新copy export与保持不变的apply/restore实现绑定到实际DSO。仅 `build_fingerprint` 字符串不够 |
| root在124对实际DSO及其部署路径独立重哈的回执 | 原DSO不下载；明确区分“本地独立重哈metadata”和“远端重新读取二进制的回执”。不得宣称本地读过DSO或实际权重 |

若某次 collect/write 失败，其引用缺失、失败响应及当时已生成的原件均保留；不得补出一个虚构完成步骤。若日志因进程中止未结束，仍能审查已发生部分，但不能证明完整清理。

## 复用旧审查器的准确边界

旧 `review39_group_originals.py` 的完整CLI不能直接接收本诊断：新顶层 schema 为 `demoncat.output39.fixed_c3_refusal_probe.v1`，它没有旧入口的 `standard_worker` 字段；source集合增加了实验文件，RPC集合增加 `install_diagnostic/finish_diagnostic`。旧检查会拒绝这些差别。

处理方式是**新增只读审核适配**：原件不变，先独立审核这两个 helper 的全部原参数、调用顺序、源身份、真实公共ticket→内部provider ticket、实际thread以及生成文件SHA；然后将明确核验过的helper分区与原Core RPC分区分别记账。完整集合必须无遗漏、无重叠，未知方法仍拒绝。可复用旧reader的 `original_join/check_group/check_live_proof/check_failed_native/final_cleanup` 等纯函数，不能篡改schema、补写 `standard_worker=true`、丢弃两个RPC，或仅忽略旧检查的issue。

新入口来源须按 source101 的实际 driver/helper、当前manifest及新Native DSO明确核验，不再套用旧C1入口的字段集合。实际标准Worker事实来自原 setup、相同进程/模型身份和真实公共请求链，不来自补上的布尔值。

## 核对顺序

1. **引用闭合。** 严格JSON解析拒绝重复键、NaN/Inf字面量、路径越界；重新计算每个原件SHA与大小。result引用、RPC有序序号、所有group originals引用和内容必须一致。较早 `actual_requests` 必须是较晚原件的完全相同前缀，不能用最终汇总替代未读取的历史原件。
2. **生命周期和真实请求。** 核对同一Worker的PID/process-start/model-instance/rank/device、public attach nonce/tenant/owner/generation和其内部provider owner映射。每个请求必须由实际enqueue→advance→执行返回/实际异常→query/collect或abort→恢复关联。三臂的完整input SHA、token ids、positions、新prefill/无prefix复用、模型权重/源码保持一致；request ID不可跨臂复用。
3. **全量分母及前置。** 从两次健康 inventory重建113个实际 projection 与28个稳定合格目标，前两个合格对象恰为预注册第0/第1层位置。完整原6前置、C2 lossless、28次独立F、OFF/health必须保留。不得只审首个C2 F后借用旧source84 C2或其他进程proof。
4. **Native事件重建。** 对原件中的 Native lease 回执按 `(Worker生命周期, Native owner, Native generation, handle)` 分组，再按 event.sequence 重建。相同事件出现在多个快照只去重完全相同副本；同序号不同内容、非前缀演进、漏open/重复apply/异常遗漏均为issue。不能把重复快照计为多次写入，也不能只读5条窄观测或 `hits` 总数。
5. **单前序F。** 在该实际请求的全部 Native 事件中，恰有一个 `mode=1, indices=[0], return_code=0, changed_elements=1` 的apply；其真实对象是第0层目标，完整 original 有限。第1层目标仅真实产生和被下游消费，没有另一个故障apply。完整余下调用/ledger证明未遗漏其他干预。对应fresh snapshot必须在该实际第1层 producer返回后、admission前，读同一对象完整span。
6. **累计F及拒绝。** 第一轮C3请求的完整已执行前缀包含恰一次上述成功写入，随后第1层同一original lease上真实apply返回4；partial inventory与失败enqueue、target ID、geometry、Native handle/owner/generation/thread、open/apply事件精确关联。未走到的调用保留为未执行，不从计划表捏造命中。拒绝目标的consumer明确未到达；前序目标必须有真实同存储值读取。
7. **live proof历史。** C2实际issued proof的内容、hash、进程内object binding、owner/scope/目标全集，与C3 consumed proof完全相同；C3 generation恰+1。该历史关联成立也只说明原对象被消费，不意味着失败C3完成，不赋予序列化JSON任何live authority。
8. **数值反事实。** 狭窄reader的五条原观测通过后，还要把每条与上述完整Native/RPC原件连接：健康第1层有限；单前序F第1层fresh span与累计拒绝lease.original完整SHA相同且均非有限；两个前序原快照SHA相同。索引/位模式/分类/完整span数目都一致。任何缺失或SHA不一致均保留为未闭合，不以相同max/数量替代完整SHA。
9. **恢复和实验清理。** 每个实际Native租约独立确认restore/release/live，首个失败事件不可被后续恢复覆盖；同时检查模型级OFF及health各自与正确的baseline摘要域精确一致。`hidden+logits bundle` 与 `logits-only` 不跨域比较。公共unregister/detach、Native零租约、无pending请求/自有hooks、原Core profile清理、实验trace和Worker属性删除分别核对。诊断写入失败可不阻止Native恢复，但不得把实验清理也标成完成。注销后普通token/text健康不能替代上述完整恢复证据。

## 允许的结果与停止边界

只有原完整请求/Native/来源/恢复检查与五条数值证据共同闭合时，才可写：“在本次固定输入和新构建下，单个前序F足以产生与累计拒绝位置相同的非有限原快照，guard拒绝与该传播一致。”这不是每个中间算子的逐条根因证明，也不能重建历史 source84 未保存的完整原数组。

可能的独立结论保留为：guard复现并完成恢复；guard复现但因果或清理未闭合；预注册位置未复现并诊断停止；观测/预算/外部中断导致失败。每一种都保留原attempt退出与原错误。当前固定全目标C3没有完成第二轮every-match/OFF验收，故**C3仍false，E5仍false**；已完成C2的原件一致性可单列，离线审核不签发新资格。

本清单不授权新NPU重跑、不扩大位置或输入搜索，也不要求调小阈值、跳过guard或修改当前运行代码。

## 已实现的离线入口与CPU合同

`review39_refusal_originals.py` 接收完整原件，不导入 observer、provider、Torch 或 vLLM。它保留实际新 schema，并审核两个 helper 后再调用旧 group reader 的原件检查函数。RPC helper分区没有授予任何例外执行权限；未知RPC依旧拒绝。旧 reader 仅补充明确 CPU ABI double 的设备表示：`device=cpu`、open.device为null，绝不作为 natural NPU接受。

运行示例（所有路径指向冻结原件和审核源码；`--evidence-root` 是该case的 `/evidence` 映射根）：

```sh
python /review-source/experiments/worker_core_v11_20261118/review39_refusal_originals.py \
  --evidence-root /original-attempt/case \
  --result probe/result.json --result-sha256 <external-result-sha256> \
  --source-root /original-source101 \
  --source-manifest-sha256 6c6b544f1709f6235961858d175bcffd6494ba47b7bbc556cf50cca49ebd715c \
  --source-commit 417d43ccc85b58554d243791546f9d484fdcd8f1 \
  --native-sha256 9261d0867f822c90d0d179457e41bc9a2b132bc39e643b61fe5d47a3fd81424f
```

JSON写到新的审核目录，不能覆盖原件。`consistent` 表示已检查原件之间的一致性，`counterfactual_pattern_and_original_joins_consistent` 表示五条数值观测与完整请求/Native链的对应关系；两者都不签发资格。原始失败C3、C2原件已完成部分、五条观测、Native分母和恢复分别报告。若helper缺失，仍保留能审查的Core完成部分，但整个审查不通过。

Native SDK/编译器/镜像/最终DSO的独立构建链必须另行审查。此入口验证source101原manifest全部依赖、loaded source回执、实际copy导出地址回执与显式DSO SHA、原六个Native函数体固定SHA，**不声称在本地重新读过远端DSO或权重**。跨Python版本的marshal代码SHA仅保留原身份并校验其来源文件、类/函数名和行号，不伪称重新编译出了相同marshal值。

本地已执行：

```text
python -B -m unittest test_output39_refusal_original_review \
  test_output39_refusal_diagnostic test_worker_core_group_original_review -q
46 tests passed; 0 skipped.
```

新增12个测试使用实际公共CPU driver、原RPC recorder、原C2→C3协议和明确的CPU Native ABI double。正例重建10个唯一lease（重复快照不增计）、单前序 `[0]` 与累计 `[0,4]`；负例覆盖额外写入、丢target/call、原文件字节变化、相同序号事件冲突、假的相同span SHA、未知/缺失helper、scope/ticket/thread变化以及OFF和资源残留。它们是审核器合同测试，**不是新NPU验收**；本次source101实际结果待原件独立审查后补充。
