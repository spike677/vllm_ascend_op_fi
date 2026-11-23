# V4 D批执行单：只观测门禁 → 前五项 → Batch/41 → EP combine

由负责人转交，准备于2026-11-23，供下一窗口使用；今晚Qwen TP2队列不因此改动。r7.1仅增加15，**没有开放DeepSeek产品**。本单的只观测/标定部分可用现有工具执行；注入部分须先收到与实际site和源码绑定的D批候选包，不能把Qwen profile改模型名后运行。

## D0：组合、制品与基线

对象固定为DeepSeek V4-Flash-0731 W8A8 / A3 / v0.32.0rc1-a3 / 实际CANN9.2.1 / TP8+EP / DP1 / 8卡专用实例。不与V4.1-1017、W4A8诊断、Qwen0.26/CANN9.1混用；不安排蓝区NPU或TP1实测。

- 记录每rank的芯片原名、逻辑设备、Worker pid/thread、TP/EP/DP、实际模型类及module、源码SHA、软件版本、已映射CANN库SHA及版本来源。模型config/量化描述/所有权重索引SHA与许可范围绑定；未知null，不补默认值。
- eager实际decode须有运行证据；无prefix cache/MTP/KV connector，单chunk，max_model_len=max_num_batched_tokens=4096，普通项max_num_seqs=1；真实prompt≥2500 tokens，加4生成token≤4096；greedy/seed固定。实际调度轨迹证明单chunk，不以启动参数替代。
- 每条prompt和canary单独标定：两个独立Worker组各至少5次；完整hidden/logits、4步token、top5留黄区。冻结两两最大差×1.5的上界；校准/normal/off/health全有限、token稳定；零上界不加epsilon。详见[dsv4_noise_acceptance](dsv4_noise_acceptance.md)。故障数值可非有限，明确记录，不能污染校准。
- 如果某rank没有可观测的完整logits，现有capture会拒绝；记录缺失的真实边界，先适配，不复制driver数组冒充该rank。不继续使用只有logprob概要的旧标定。

## D1：只读采集、源码及文本回传（可先执行）

工具使用独立源码目录D_SOURCE，固定Git提交`ef53263bdf9b9139ce92c4c8eb27bbb5e98c124d`；不要假设r7.1源码tar包含D探针。可从该提交的完整仓库取得，或使用已交付的独立工具补充件并记录其manifest及脚本SHA。实际运行前核对下列文件均存在；缺文件停止，不换未知旧脚本。

```sh
cd "$D_SOURCE"
sha256sum experiments/worker_core_v11_20261118/dsv4_noise_capture.py   experiments/worker_core_v11_20261118/dsv4_observe_probe.py   fault_injection/sites/dsv4_observer.py fault_injection/dsv4_noise.py   tools/dsv4_capture_bundle.py tools/review_dsv4_noise.py tools/extract_dsv4_record.py
```

可信宿主的具名RPC包装仅转发以下函数（不能开放任意代码给普通用户），源码paths取[dsv4_0731_source_review_20261123.json](dsv4_0731_source_review_20261123.json)，不得用候选SHA填实际结果：

```python
from experiments.worker_core_v11_20261118.dsv4_observe_probe import observe
from experiments.worker_core_v11_20261118.dsv4_noise_capture import capture
# 在已有Worker对象上，受信线程调用；不要新造Worker。
# source_rows = observe(worker, 'source', {'paths': reviewed_relative_paths})
# observe(worker, 'start'); 真实HTTP请求完成后 rows = observe(worker, 'stop')
# capture(worker, 'start', phase='calibration')
# 真实HTTP请求完成后 numeric = capture(worker, 'stop')
# 异常则capture(worker, 'abort')，记录清理结果。
```

观察与标定分窗口，避免两个工具争用已有边界。全rank各收一次，不用rank0乘8。observe逐项报告prefill/decode/unknown、真实operator/path/shape/dtype、调用请求归属、不可观测与不适用原因；source路径字节用Linux Git blob身份对比，不修改文件换行。

```sh
# 每个真实校准样本：envelope含capture(8rank)、tokens、prompt_token_ids、
# single_chunk、sample_id、process_identity，均来自实际运行。
PYTHONPATH="$D_SOURCE" python -m tools.dsv4_capture_bundle "$PRIVATE_ENVELOPE" > "$PRIVATE_RUN"
# 宿主合并真实combination/calibration；正式四阶段另加phases。
PYTHONPATH="$D_SOURCE" python tools/review_dsv4_noise.py "$PRIVATE_INPUT" --output "$PRIVATE_SUMMARY"
PYTHONPATH="$D_SOURCE" python tools/extract_dsv4_record.py noise "$PRIVATE_SUMMARY" > "$NOISE_TEXT"
PYTHONPATH="$D_SOURCE" python tools/extract_dsv4_record.py observe "$PRIVATE_OBSERVE" > "$OBSERVE_TEXT"
```

输出每页≤4KiB，原样回传，原summary/数组/回执留黄区，只报文件名/字节/SHA。每条prompt分开，不截字段。脚本与执行回执的版本要一致；旧工具的候选eligible只是类型线索，不代表注入已适配。

## D2：按实际调用修正site，顺序固定

|顺序/机制|必须证明的真实边界|不允许的替代|
|---|---|---|
|1 28|V4-0731实际dsa_attn_kv_plan/SparseAttnSharedkv及Compressor，真实稀疏索引、压缩KV、mask与请求边界；分母重放块与合法key集合|不能套V2-Lite MLA或旧V4.1 SparseFlashMla/LightningIndexer名字；padding不得当合法key|
|2 18|hc_pre中的RmsNormCast及HcPre/HcPost真实值消费者，mHC展开维度|不能仅按旧RMSNorm类匹配后宣称覆盖|
|3 22|indexer真实InplacePartialRotaryMul、positions/频率/原地范围和后续Q/K消费者|不能把Qwen融合mRoPE的partial规则直接套入|
|4 39op|MatMulV2/QuantMatmulWeightNz等实际BF16 ND投影输出，真实消费者；新增Inf与已有Inf分记|不改变累加器，不接受未知dtype/layout|
|5 39mr|实际权重/量化scale/bias/NZ布局与投影输入join，正常完整范围控制、命名错误累加规则|不能仅靠算子名推测Python调用覆盖或事后乘输出|
|6 Batch|独立DynamicQuant/V2真实Q/scale签名，QMM确实消费；当前decode批次至少2真实请求|GroupedMatmul内部融合量化不覆盖，B1必须不注入|
|7 41|Compressor/slot_mapping/state_block_table实际压缩KV写入区、消费者和安全恢复点|不能套Qwen ScatterPaKvCache页布局；不使用CANN9.1 Native库|
|8 8|MC2 TokenDispatcherWithMC2.token_combine真实输出行→本轮请求/token的动态join和消费者|不改expert_ids、ep_send_counts、assist_info、通信组来代替行错位|

D0/D1文本审核通过后才对每项锁定源码/结构/部署范围和命名变体，交独立D候选包；未适配项保持关闭。Batch因≥2500token长输入合批，8192或更高token预算不能从Qwen直接借用：按实际并发数与长度另作磁盘/显存/单chunk预算及许可登记，先B2；无合格资源则记录部署不适用，不能违反4096当前已声明范围。

41 Native必须用9.2.1新组合的构建收据/编译器/源码/库SHA及实际映射库身份审查；Qwen9.1产物不可复用。安全恢复点不明就拒绝。

## D3：候选后的统一产品验收（当前不能提前执行）

候选包必须同时给出受信profile、许可source-review SHA、范围、噪声金丝雀接线和完整效果审查/提取工具；r7.1的Qwen精确canary不能通过修改阈值冒充D产品支持。

同一Worker组两个owner。宿主仍仅attach/enable/status/disable/detach，Worker自动发现真实请求；mode=all，全rank每次合格prefill/decode持续触发，4生成token。开启期间三个不同真实用户请求；Batch用真实并发B2起并加B1负例。CLI使用`dcatpfi fault ... --socket`，显式机制ID沿用公共命名；不能在D候选尚未接线时执行enable。

每owner先运行冻结噪声规则下的normal/canary，随后fault真实请求；disable排空后off/health新请求。每rank逐项报observed/eligible/attempted/applied/changed/consumed/restored/failed，eligible失败0，changed≤applied（写入相同不是失败），真实设备消费完成，区域字节精确恢复、活动资源零、off/health token相同且数值在冻结上界内。没有调用的EP rank记录0及原因，不制造调用，不在条件命中点通信。

同源same-prompt的normal/fault/off/health数值对照与三条用户请求覆盖分开保存；fault超噪声或token改变是模型效果，未超界只报告site效果。不能用同一组Worker第二owner重启掩盖状态。耗时倍数只记录，阶段900秒；预算超出按真实失败保留，不改成功字段。

负例必须有合法正对照：缺许可、未知profile、CLI99、身份漂移/跨TP借用、超长度/5token、Batch B1、布局/源码不匹配；失败后确认资源零，只有缺许可时的ACTUAL_CALL待观测项不算另一个错误，其他准入不满足不可混入。

## D4：8的EP动态映射专项

逐rank、逐真实调用记录scheduler请求顺序、scheduled counts、computed token位置、输出行到scheduler行的映射，以及padding的实际索引；用`fault_injection/sites/dsv4_request_rows.py`做一一join。拒绝重复、遗漏、越界或陈旧请求集合。故障可能改变路由，不能用baseline调用清单强求fault相同。

combine通信先完整执行且参数原样透传，随后只在已确认同请求行内循环错位，padding和单行组不改；消费者在同stream实际读完后恢复。单行decode无可移动行如实记录，不作为applied；eligible失败不能当skip。V2-Lite AllGather/unpermute旧证据不授MC2资格。

## 回传及审核结论

每机制两个owner对应表、每rank提取页、实际源/库/模型身份、消费者、三项恢复、负例与正对照、各回执SHA。Codex逐份给通过/不通过/需退补，列明登记键和缺项；只观测/CPU通过不登记。D产品未适配前登记清单为空，不能用Qwen或旧V2-Lite组合填补。
