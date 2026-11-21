# DeepSeek V4.1 状态残留：只读静态分析

2026-11-21 夜。未修改第三方代码，未在蓝区加载 V4.1、运行 0.32 NPU 或下载权重。以下是候选解释及证伪方法，不是黄区不确定性的根因结论。

## 源码身份和适用边界

读取官方 vllm-ascend `v0.32.0rc1`，提交 `24487f4f48ee99cf45d86d1c504e645549b8c1ef`，源码只读副本在项目外发内容之外的 `.codex-tmp/vllm-ascend-v032-source`。上游 vLLM 依据发行文件指定的 `v0.32.0`。没有把 tag 源码等同于黄区镜像安装字节：黄区实际函数/配置分支未回传前，结论仅覆盖这份源码。文件指纹见 `dsv41_state_residual_source_review.json`。

## 1. 短序列与无效 Top-K

**源码不是完全没有屏蔽。** A3 V4.1 的实际 Python 调用在 `attention/dsa_v41.py`，不能用另一个仅用于特定 BF16/A5 路径的 `attention/sparse_flash_mla.py` 代替。

- `models/deepseek_v41/indexer.py:227`：压缩缓存最大长度为零时，稳定 indices 缓冲区填 -1，长度缓冲区清零。非零时调用真实 QLI v3，再做索引后处理。[源代码](https://github.com/vllm-project/vllm-ascend/blob/24487f4f48ee99cf45d86d1c504e645549b8c1ef/vllm_ascend/models/deepseek_v41/indexer.py#L227)
- `ops/triton/prepare_indexer_indices.py:48`：按 query position 和压缩比计算可见长度；负数或超出可见范围的索引改为哨兵，排序后写 -1，并可写有效长度。真实输出范围内每个索引槽都执行 store。[源代码](https://github.com/vllm-project/vllm-ascend/blob/24487f4f48ee99cf45d86d1c504e645549b8c1ef/vllm_ascend/ops/triton/prepare_indexer_indices.py#L48)
- `attention/dsa_v41.py:694`：压缩索引补齐为固定 Top-K，padding 为 -1；向 SMLA 传递实际 KV 长度、压缩余数、mask_mode 和 block table。[源代码](https://github.com/vllm-project/vllm-ascend/blob/24487f4f48ee99cf45d86d1c504e645549b8c1ef/vllm_ascend/attention/dsa_v41.py#L694)

仍需验证的候选：QLI 没有写满的槽若残留一个**恰好仍在合法范围内**的旧索引，单靠范围过滤无法识别其来源；原生 SMLA 对 -1、压缩长度和尾块的实际读取也不能由 Python 参数传递单独证明。下一次只读对照应同时保留原始 selected、后处理 indices、每行可见长度、实际有效 key 集合及重复索引计数，比较短→短、长→短、重复短请求。若输入集合/有效长度均相同而输出仍变，继续区分原生归约与状态读取，不能直接归为“未 mask”。这些是给负责人选择的验证方法，未自行要求黄区操作。

## 2. KV 分配、压缩状态与复用

源码有新分配块清理路径：Ascend MRV1 根据 scheduler 的 `new_block_ids_to_zero` 调用 `_zero_block_ids`；后者委托上游 zeroer 并单独恢复静态量化 scale。MRV2 的 zeroer 将 tuple/list cache 拆成组件，跳过 runner-only 层和空 view，对列出的块逐组件清零。[MRV1](https://github.com/vllm-project/vllm-ascend/blob/24487f4f48ee99cf45d86d1c504e645549b8c1ef/vllm_ascend/worker/v1/model_runner.py#L1024)、[MRV2](https://github.com/vllm-project/vllm-ascend/blob/24487f4f48ee99cf45d86d1c504e645549b8c1ef/vllm_ascend/worker/v2/utils.py#L24)、[上游调用条件](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu_model_runner.py#L1201)。

**有清理函数不等于每次请求的全部持久状态都清零。** 清理依赖实际 scheduler 块列表、cache group/组件枚举和布局映射；压缩器工作区、shared Top-K/candidate 缓冲区不自动等于 KV 页。正常实现也允许不清除有效长度之外的旧字节，只要算子确实不读取。

候选验证：按实际 runner 版本记录新分配/复用块 ID、zeroer 覆盖的 cache 组件和写入区间，核对当前 query 合法读取区间是否全部由本请求写入或合法前缀提供。只观察被消费区域，不因发现未读尾部非零就宣称故障。长→短敏感但短→长稳定会支持状态/边界候选，仍须定位首个受影响的真实输入或输出。

## 3. Engram host gather / offload

当前 tag 的 `cpu_offload` 主路径不是“CPU gather 后 H2D”：`HostUvaBuffer` 通过 ACL 注册 host 内存，NPU kernel 直接读取设备可寻址的 host 表；有 64KiB 对齐及释放前设备同步。另有 `_torch_lookup` 作为 CPU 参考/bring-up 路径。[UVA](https://github.com/vllm-project/vllm-ascend/blob/24487f4f48ee99cf45d86d1c504e645549b8c1ef/vllm_ascend/models/deepseek_v41/engram/npu.py#L156)、[分派及 CPU 参考](https://github.com/vllm-project/vllm-ascend/blob/24487f4f48ee99cf45d86d1c504e645549b8c1ef/vllm_ascend/models/deepseek_v41/engram/embedding.py#L381)。

两个 gather kernel 都将不属于本 shard 的 ID 映射到安全行，并将对应输出写零；实际 `lookup` 将 local_heads 与 pad_heads 都设置为本分片头数，不能仅根据 kernel 注释“padding 未写”认定真实调用漏写。模型还等待 lookup ready event，并在输入复用前 join prepare stream，失败时重置旧事件。[kernel](https://github.com/vllm-project/vllm-ascend/blob/24487f4f48ee99cf45d86d1c504e645549b8c1ef/vllm_ascend/ops/triton/engram_lookup.py#L45)、[退役顺序](https://github.com/vllm-project/vllm-ascend/blob/24487f4f48ee99cf45d86d1c504e645549b8c1ef/vllm_ascend/models/deepseek_v41/model.py#L1329)。

候选验证：先确认黄区实际走 UVA、设备表还是 CPU 参考；固定 hash/lookup ID、分片范围和有效 token 数，比较 gather 后完整有效行以及消费者所等事件。若 ID 已变，优先检查 lookback/request 映射；若 ID/表内容相同而 lookup 值变化，才聚焦映射可见性、事件与持久输出缓冲区。源码中的保护措施尚不能代替黄区实际执行证明。

## 已有黄区证据与待对照项

已接收材料显示 DeepSeek 同进程也出现第4/8 token 差异，Qwen A3 的首批 B8 对照稳定。它们不能证明 DeepSeek 是跨进程初始化问题，也不能把 Qwen 的稳定性外推给 MLA/Engram。今晚新对照结果尚未到达；没有用默认配置补齐 runner、mask、offload 或确定性设置。收到后优先补充“实验条件→候选支持/反证”的表格，仍以黄区实测身份为准。

## 备选验收设计（未启用，待负责人决定）

|方案|前提及比较方法|不能掩盖的失败|成本|
|---|---|---|---|
|D1 长 prompt|只有证据确认短序列/尾部读取条件后，按各层实际压缩比和 Top-K 确定阈值，继续完整精确比对|长 prefill 的早期 query 仍短于 Top-K，因此仅增加总长度不能保证避开问题；需证明早期行也正确，或明确限定缺陷只发生于 decode|增加 prefill 开销，实测后给数值|
|D2 同位置双进程对照|A跑 normal×4；B跑 normal/fault/off/health；相同确定性配置、请求/token、预热、调度和分配条件，按第1～4位置比较；至少重复两套|B故障改变持久状态而导致第3/4位置与A不一致，应判恢复失败/未证明，不能当自然噪声扣除。A自己不稳定也不能授予精确通过|两个启动成本+八请求；两个作用域需明确共享或重建方式|
|D3 每阶段新进程|每阶段从相同干净输入启动，比较基线与故障及新OFF/health|它证明进程重建后的恢复，不等价于同Worker关闭/恢复；不能未经批准替代当前两作用域标准|两作用域至少8次启动，约8×(启动+预热)+8次请求；若启动10分钟则仅启动约80分钟，属于条件估算，不是V4.1实测|

当前仍采用确定性条件下的完整精确比对。未启用噪声阈值、D1、D2 或 D3，也未修改现役或第三方代码。
