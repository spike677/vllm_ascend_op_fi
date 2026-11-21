# 28 denominator replay torch：真实模型持续接线验收

## 结论范围

V2-Lite W8A8 与社区 Qwen3.8 W8A8 均已在 124 A2、0.26、TP2 完成真实模型接线验收；两者均已在独立重启 a4 中通过新增字节级恢复复核。本页证明可信实验 binding 下的真实执行与恢复，尚不授予公共 Runtime/RC3 组合准入。最终 a4 汇总见 `experiments/worker_core_v11_20261118/continuous28_real_model_summary_20261121.json`。

## V2-Lite TP2

- 实际路径：27 个 `AscendMLAImpl`；每 rank 8 个本地 query head。prefill 是展开 Q/K/V，decode 是吸收 Q + 分页 latent/rope cache，随后调用真实 `_v_up_proj` 与 `o_proj`。
- 输入：固定短文本，实际 prefill 10 tokens，生成 4 tokens，即 prefill 一次、decode 三次；一次一个实际请求。不能外推多请求混合批、长上下文、稀疏 MLA、EP 或图模式。
- 每个调用选取该请求最后一个 query 行；每 rank 每层每次真实调用均执行。块大小 128、首块分母贡献重复一次，分子不重复；不是合法重排或历史 kernel 根因复刻。
- a1 和独立重启 a2 均完成同一组 Worker 两个独立 owner 的 normal/fault/off/health。a2 增加 pause/enqueue 得到真实内部 ID，再安装观察边界、恢复调度，确认两个 rank 的四个 forward 都属于该 ID，最终输出映射回外部 ID。

|a2 每个作用域、每个 rank|observed|eligible|attempted|applied|changed|consumed|restored|failed|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|rank0|108|108|108|108|108|108|108|0|
|rank1|108|108|108|108|108|108|108|0|

108 = 27 prefill + 81 decode；两个作用域合计 432 次实际改写。非目标行检查指同一次调用改写前后的比较，不是说连续传播之后 B/F 的全部其他层/行仍相等。

消费者 pre-hook 在实际 o_proj 输入中逐元素核对替换行；post-hook 在同 stream 记录并等待设备 event，确认投影读取完成后恢复注入器拥有的临时区域。保持强引用防止分配生命周期结束或地址复用；同一 site 上未完成租约禁止再次进入。

两个 rank 的完整 hidden/logits 在 fault 均变化；OFF/health 四个 forward 的完整张量 SHA 与 normal 相同。两组 fault token 也变化，但 token 变化不是验收门槛。正常短输出不构成完整语言质量验证。

- a1 故障/正常耗时：1.8669×、1.5850×；a2：1.7082×、1.6056×。
- 这些是带完整输出观测开销的短请求耗时倍数，不是无观测生产服务性能。建议本短请求实验单次上限 30 秒；扩大长度/批大小前重新测量，不据此开放任意长度。
- 模型权重、cache、真实输入未通过注入器重新绑定；恢复的是自有替换区域，OFF/health 是新的干净请求，不承诺将 fault 请求逆转到未注入轨迹。

## 源码与独立复核

- `fault_injection/engines/attention.py`：FP32 online max 重标定及分母重放。
- `fault_injection/sites/mla28.py`：真实 prefill/decode 边界、分页 latent 解包、V-up、消费者和设备完成证据。
- `fault_injection/sites/gqa28.py`：Qwen full attention 与 gate/o_proj 接线，另行实测。
- `experiments/worker_core_v11_20261118/continuous28_mla_probe.py`：标准 vLLM TP2 Worker，两作用域、四 token、预热与完整输出恢复检查。
- `tools/review_continuous28_site.py`：从保存的回执重新核对每层 prefill/decode 次数、消费、恢复、请求映射、同 Worker 与独立 owner。
- 复核工具负向测试覆盖遗漏消费者、恢复失败、错误请求 ID、OFF 差异、缺 rank 和重复 prefill；连同参考重算回归首轮 7 项 CPU 测试通过；新增字节恢复负向用例后为8项。

MLA 实际安装源码 SHA256：`67b4479cca16fd164b33b91b2fa4f48e95f2474720e484abdf3b3897a8ea1545`。
a2 实验 source manifest SHA256：`a992fc7c5cceccdff81be062ac7c7c0e086b20e70904847d82227ca39ada6a9c`。
镜像固定为 `sha256:c9c9432b69767cf1855efad44f20eaa8ab02520ae05b75c936e67ea4e7eb2702`，沿用已核查的 vLLM 0.26.0+empty / Ascend 0.26.0rc1 / CANN 9.1.0 组合。

原始回执和日志留在实验目录；不发布完整私有张量，不提交新 statistics 数据。公开材料以实现、汇总和源码指纹为主。

## Qwen3.8 TP2

修复后的 a3 通过独立回执复核。64 层模型中 16 个 full attention 层全部覆盖，48 个 GDN 层不属于本能力适用边界，未注入。prefill 为 11 tokens，生成 4 tokens；每个实际调用改写目标请求最后 query 行。

|a3 每个作用域、每个 rank|observed|eligible|attempted|applied|changed|consumed|restored|failed|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|rank0|64|64|64|64|64|64|64|0|
|rank1|64|64|64|64|64|64|64|0|

64 = 16 prefill + 48 decode；两个作用域合计 256 次改写。实际 gate 从真实 QKV 投影输出取得；o_proj 输入逐元素等于故障 Attention 行经过该 gate 的结果。正常输出缓冲区与自有替换缓冲区都在真实消费者完成后恢复，非目标行保持本次改写前原值。

两个 rank 的 hidden、logits 均变化；两个作用域的 OFF/health 完整四次 forward 输出与 normal 精确相同，资源计数为零，同组 Worker 身份保持不变。故障/正常请求耗时为 1.2553×、1.1762×；故障实际耗时 1.462s、1.343s。模型正常 token 为 `[271,760,12515,7701]`，故障为 `[271,248068,198,1421]`，两个作用域一致。这不构成完整语言质量或长文本验收。

a2 失败保留：第一次在消费者校验中将 `[heads,dim]` 与扁平 gate 相乘，出现 256 对 3072 维度不匹配。这是本适配器缺陷；修正为按真实投影输入顺序展平后，新 a3 全矩阵通过。未改模型、权重、宿主软件栈或故障公式。

a3 source manifest SHA256：`24c92eae5c1ab765ec72ec621cf05a9c7414251cd76354f13ea0b376b9d494b7`。
Qwen 实际 Attention 源码 SHA256：`4ba903ee8afb23012e78870034b1280039ccf81edb41b8e16f00e6e5bbc4c076`。
回执 SHA256：`7b1619e232b140c22580ac687f7d4b7163a37f8fb48084926df54eb37a8a5e8d`。
实验脚本的历史完成状态字符串含 `MLA`，Qwen 本次实际使用 `--kind gqa` 和 GQA site；不据该字符串将 Qwen 分类为 MLA。

模型配置指纹：V2-Lite `229913ea2a346ccdb571bb3cb23414ca3a0ee1a9455fe88a15a5788bc837cb75`；Qwen `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`。相同 config 不代表社区与黄区量化制品相同。

## 资源窗口

以下时间均为北京时间 2026-11-21，时长包含初始化、加载和清理，不是纯 kernel 时间。

|窗口|开始|结束|耗时|结果|
|---|---|---|---:|---|
|a1 V2-Lite|17:07:53|17:12:47|294.154s|两作用域通过|
|a2 V2-Lite + Qwen|17:13:52|17:20:45|413.317s|V2-Lite复核通过；Qwen消费者布局错误|
|a3 Qwen|17:22:12|17:26:17|245.091s|修复后两作用域通过|
|a4 V2-Lite + Qwen|17:29:24|17:36:19|415.078s|两模型独立重启，逐调用字节恢复复核通过|

总卡窗口约 1367.640s。四个窗口均使用原资源锁、每 case 门禁和只针对卡2/5的有界清理器，结束后清理器停止并 join；实际没有停止外部容器。未改 Docker 入口、驱动或 CANN。最终只读检查卡2/5无 NPU 计算进程。

## a4 最终严格复核

不改变公式、输入或命中范围，给每个调用增加写前与恢复后的原始字节 SHA256。独立工具逐条要求二者一致；Qwen 还要求实际 output buffer 恢复 SHA 一致。不能用 `restored=true` 替代这项证据。两模型在同组 Worker 的两个作用域均通过，覆盖数仍分别为每 rank/每作用域108和64，failed为0；consumer、非目标行、内部请求映射、全 hidden/logits OFF/health 精确恢复及资源清零均通过。

最终故障/正常耗时倍数：V2-Lite **1.7266× / 1.6460×**；Qwen **1.2898× / 1.2172×**。两个模型两个作用域的 token 都发生变化，但不作为验收要求。当前验收只涵盖这份短提示、prefill加三次decode和每调用最后query行。

- a4 source manifest：`d8661574b949140dd2093ed4dcae36ad197df7b21700f9c60cd83220ec909c46`。
- V2-Lite回执：`dd2645d1c18784b5ea302e5793d60bbe4b440e2de12af5b8a4bbb6d412fdf2bf`。
- Qwen回执：`bc2660eb6a8fa29533417616a50a912853ce8cf8bd8a8b8ccd86a1ead8feeb56`。

vLLM退出时仍报告Worker宽限期后SIGTERM和一个shared_memory清理警告，已保留日志。注入器owned hook/lease/inflight清零是在活着的Worker内核验；进程退出告警不能替代或否定这项资源检查，但也不应称宿主退出完全无告警。

## 未完成交付项

本次是实际模型 site 接线和效果验收。现有公共 Worker Runtime 的可信 typed binding、持续 request adapter、最终 coverage/recovery 收尾准入仍需接入并验收，不能因为实验 RPC 已跑通而将公共创建入口放行。稀疏 MLA、更多请求、更多形状和 TP8+EP 未被本次覆盖。
