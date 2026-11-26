# N2 无 tiling 边界依赖的离线变体

选择交付 `tools/extract_fault33_launch_host.py`，schema 为 `dcat.fault33.n2.launch_host.v1`。**完成 N2 launch/host 证据运输，不等于完成 tiling 解码，更不放行控制重算或故障33。** 不改旧 N2 的 tiling 门禁，不改冻结包、采集器或推理进程；无需 NPU、无需重启，不作为 D/G 前置条件。

现有80字节 `<10Q` header 可以核对 pid、serial、launch key、blockDim、host 字节数、holder 数量/宽度及文件总长度。由它能证明的是 `[80,80+host_bytes)` 完整 host 区间。holder 中的两个32位数及地址关系，本身不能证明哪个对象是 tiling、对象长度或 CANN 模板布局；不能把下一指针或 host 尾部当作 tiling 边界。当前尚无足够边界证据，因此 tiling 一律 null。

## 执行（已有六条原件，不补采）

使用 Python3.10+ 标准库。在本次开发源码根目录设置 PYTHONPATH；依赖只有同目录 `extract_fault33_existing.py` 和 `review_launch_capture_v2.py`，无 torch/ACL 依赖。review_launch_capture_v2.py 与 r8 文件不变。

```bash
export PYTHONPATH=/本次开发源码根目录
python "$PYTHONPATH/tools/extract_fault33_launch_host.py" \
  --capture-root /现有三窗口原件的共同根目录 \
  --api /现有真实API字段.json \
  --plan /现有n2-map.json
```

工具只向 stdout 输出 JSONL，不改输入状态。计划继续使用六条（128/512/2304 × rank0/1），`source_sha256` 仍为已审黄区版本 `1fecdc87bb4ea3f923d25d9d57e4bad83da41060d7c986bdc1c2635b0b24a4dc`。每条 `capture` 指向真实 bin；`pointers` 是 JSON Pointer，必须指向 API 文件的真实键，不能照搬别人的层级。每条 `tiling` 缺省或 null；禁止填猜测的 holder_index/bytes。

必须核对实际 rank、prompt_length、api、module、phase、q_shape 以及记录的唯一 `launch_relation`（pid/serial/unique/method）。不能用 serial 相邻推导关联。blockDim 来自 header，不要求 API JSON 中杜撰此键。block_table 有效范围、mask 等尚未记录的字段保留 null 和 deferred_issues，不把 shape/stride 解释为有效范围。

退出0表示六条 launch/host 运输检查通过；退出2表示仍有阻塞。未知源码、原件截断、六条不全、形状/rank重复、API与launch关联不唯一等不能降级忽略。若当前原件缺真正的关联字段，须如实回报阻塞，不能通过手写 API 事实来补齐。

## 回传与边界

输出顺序：summary、六条 row、若干 bytes 记录。每条含 record_index 和规范化 JSON（UTF-8、sort_keys、无空格、排除 extract_sha256 自身）的 SHA256。每行含换行不超过4096字节；超限报错，不截断。按原顺序分段粘贴，保留所有行。

bytes 记录把 **完整 host** 和 **opaque holder 表** 分别按768字节切块，含 capture SHA、region、offset、total_bytes、region SHA、chunk SHA、base64数据。接收方逐块校验 SHA；按 row/region 分组，验证 offset 连续无重复，base64解码拼接后长度等于 total_bytes、SHA等于 region_sha256。这样可在蓝区做候选 ABI/布局分析，黄区无需猜测 tiling。

summary 固定保持 `field_extraction_complete=false`、`serializer_validated=false`、`ready_for_control_recompute=false`、`qualification_granted=false`。`n2_launch_complete=true` 不能被登记器当作33通过。launch key 仅为已审文本差异来源的候选身份，不宣称完整采集源码审计已完成。

后续仍需公开源码候选身份、字段边界、三组形状交叉验证、每组控制重算逐位一致率≥99%并定位差异；任何一项不满足就停止。该变体只解除“必须先知道 tiling 边界才可传完整 host”的循环依赖。

CPU 测试覆盖六条、缺字段保留 null、假关联拒绝、两条代六条拒绝、手填 tiling 拒绝、完整数据与 SHA 重建，以及65536字节最大 host 的分块/大小限制。黄区实际六条运行结果仍待回传。
