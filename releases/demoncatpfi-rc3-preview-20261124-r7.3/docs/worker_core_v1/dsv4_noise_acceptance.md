# V4-0731噪声范围验收规则 v1

规则固定为`demoncat.dsv4_0731.noise.v1`，安全系数**1.5**，不得看fault/off结果后调整。仅适用于A3 / V4-Flash-0731-W8A8 / vllm-ascend0.32.0rc1 / CANN9.2.1 / TP8+EP / DP1 / 8卡 / 上下文≤4096 / 单chunk。量化描述、全部权重索引、config、运行源码身份必须完整。不是V4.1，也不适用于Qwen；Qwen保留逐位精确恢复。

## 标定

每条验收prompt及产品canary各自标定，不跨prompt借阈值。实际token长度≥2500且加4生成token不超过4096。固定temperature=0、生成4token、单chunk；同一专用进程K≥5次，新进程再K≥5次。记录实际process身份、唯一sample ID、prompt token SHA与长度、四步生成token、各rank四步完整hidden/logits和top5 logprob。请求顺序不改动为只重复同一输入的假稳定；若混合顺序属于实际范围，另作相同标定组并绑定其顺序身份。

完整数值数组仅保存在黄区私有回执，蓝区只接收总结文本。只有min/max/mean、hash或logprob不能代替完整hidden/logits的差值观测；缺失一律不可验收。

在每个rank、每一步，取两个进程全部标定样本两两比较：hidden最大绝对差、logits最大绝对差、按概率排名对齐top5 logprob最大绝对差、top5 token集合Jaccard距离。每项上界为该项最大实测距离×1.5；零上界仍为零，不加epsilon。所有标定生成token必须完全一致。top5排名logprob指标明确不是同一token的概率差，集合距离另列，完整logits差仍独立检查。

## 四阶段判定

normal/fault/off/health必须同一组Worker、同一prompt。新的normal须在预冻结标定范围内，不稳定时不扩大阈值。OFF与health：token必须与normal完全一致，每个rank/step全部数值指标不超过上界。fault：若任一数值指标超界或稳定token改变，模型效果可见；否则只报告site效果，不声称模型输出变化。

fault/off/health的区域字节恢复必须严格成立，leases/hooks/inflight均为0；噪声口径不能放松这些要求。注入点消费者、全rank持续覆盖和两作用域要求不变。当前代码是独立验收判定器，未登记V4组合、未修改Qwen或已冻结r5，不会自动授予资格。

## 黄区采集与回传步骤（负责人转交）

1. 使用上述固定专用实例；先核对已加载的CANN版本为9.2.1，而不是目录软链接推测。记录实际模型、源码、制品和参数身份。
2. 在可信验收驱动中，将`experiments/worker_core_v11_20261118/dsv4_noise_capture.py`的capture作为受信collective_rpc callable；每个单独校准请求前start，通过普通推理API发真实请求，真实完成后stop。它仅包裹实际compute_logits读取完整hidden/logits，不发请求、不写张量；任何异常用abort移除自己的观察器。此接口不能开放给普通用户。
3. 宿主将每rank的steps、真实driver生成token、实际prompt token SHA、process身份和唯一sample ID组合成run。一个run含ranks="0"…"7"、tokens四项、prompt_tokens、single_chunk=true、prompt_sha256、sample_id、process_identity。不同进程的process身份不得相同；每个进程sample ID不得重复。所有数组留在黄区，不能用统计摘要伪装完整数组。
4. 输入JSON结构为combination + calibration（≥10个run）；正式四阶段另加phases={normal,fault,off,health}。每阶段run还含实际region_bytes_restored及resources={leases:0,hooks:0,inflight:0}；这些从机制回执取值，不由观测器宣称。
5. 运行`PYTHONPATH=<source> python tools/review_dsv4_noise.py <private-input.json> --output <private-summary.json>`。它从原始标定重新计算规则，不接受外部手填阈值。每条prompt单独输出，超过60KB则拒绝文本导出，不截掉失败信息。使用tools/extract_dsv4_record.py noise生成每rank不超过4KiB的提取文本回传；原summary和完整数组留黄区，只回传文件名、字节数和SHA。
6. 先标定与只观测，不因本页自动启用V4注入。回传后核对数值有限性、token稳定、上界、源码匹配及site适用性，再决定第二批包。

待黄区标定数据到达；目前只有CPU规则测试通过，不是V4实机验收。W4A8 NaN诊断单独接收，不改变上述W8A8目标。

故障阶段允许显式记录 `+inf`、`-inf`、`nan` 字符串（JSON 不含非标准 NaN）。这些是超出有限标定范围的现象，不能解释为噪声；normal/off/health 与所有校准仍必须有限。capture 的 start 通过可信驱动指定 `phase="fault"`，其他阶段默认拒绝非有限值。若某 rank 没有可观测的 compute_logits 返回值，当前采集器明确拒绝；不得复制 driver 的数组伪装该 rank 的观测。需按实际模型边界补齐观测后再验收。

2026-11-23补充：上述capture及observe漏打包，已通过独立B组工具补充件补交，不改r5/r6标签。先验证采集→组装→review→extract链路再开始正式标定；不能用未知旧脚本的摘要替代完整数组。

11-23下午补充：新“logprob约1e-5”摘要没有定义top1/同token/排名指标，不能与旧记录其他候选的0.5～1.0直接比较。旧首选token本来就只有约1e-6～1e-5的变化。实际prompt token SHA、进程和确定性配置尚未对齐，不缩窄上界，不调整1.5系数。详见[yellow_intake_20261123_afternoon.md](yellow_intake_20261123_afternoon.md)。源码比对采用新Git原始字节清单，不用旧Windows CRLF工作树哈希；原始黄区文件与回传hash始终不改。
