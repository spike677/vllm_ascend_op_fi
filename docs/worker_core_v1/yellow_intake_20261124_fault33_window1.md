# 33 窗口一与采集器改动审查（K3）

对象：黄区 A3、Qwen3.8 W8A8、TP2、0.26 / CANN 9.1 路线。证据是负责人提供的原样文本，原始 capture、完整源码、window1-result.json 留在黄区。本记录不等于蓝区实测或 A3 准入登记。来源 SHA 见同名 JSON。

## 结论

**原始采集文件可以保留，用于离线补提，不要求现在重跑窗口一。目标 key 和 serializer 尚未审定，不能进入控制重算或放行 33。** 这替代此前因未取得 diff 而统一要求换 r2 重跑的建议。封存旧文件；r2 仅用于必要的新采集，旧记录不伪造 r2 sidecar。

可沿用的文本证据：三组实际 prompt 128/512/2304、各生成 4 token；prefill TND、Q `[N,12,256]`、K/V `[N,2,256]`、BF16、sparse_mode=3；decode Q `[1,12,256]`、K/V cache `[8532,128,512]`、block_table 非空。实际 Python 入口报告为 DIRECT_npu_fused_infer_attention_score → FIA V1。此前九库身份和同进程逐位金丝雀沿用对应原件 SHA，不用新默认值补齐。

目标只选 16 层 full attention 的真实 paged decode（Q 长度 1、实际 block_table）。GDN 和 unified_attention 是外层/其他分支观测，不拿它们的计数或参数当 FIA launch。API 的 V1/V2/V3 是入口签名版本，**不是 tiling serializer 的标识**。导出多个 API 符号也不能证明某一 serializer 被选择。

## 改动审查与限制

声明源码 SHA：`1fecdc87bb4ea3f923d25d9d57e4bad83da41060d7c986bdc1c2635b0b24a4dc`；库 SHA：`93df16e8d734c5ddb55a3e78fe8f46f1c03914defeb8a3ec39e9ec3f4627e15e`。这里只审查提供的 diff；未持有全源码，不能声称独立复算了源码或库 SHA。

1. 新增 GetFunction hook 先调用真实函数，再记录句柄和函数名哈希；未改 launch 参数、host 字节、返回值或设备内容。Worker 扩展文字描述为只读 API census、logits 哈希和 flush，与此目的相符；未收到其完整实现，不额外推导隐藏行为。
2. 按名路径用高位 1 的哈希占位。**它不是 tiling key**。unknown 降为 0 只说明句柄可查，不证明真实 key 可查。
3. ByEntry 覆盖高位占位，但已有低位 entry 不再更新。若句柄随后被不同低位 entry 复用，记录可能过时；高位真实 entry 也不能与哈希区分。因此旧记录只把非零低位值列为“文本 diff 支持的 ByEntry 候选”，不能仅据此锁定分支；须补唯一 API/launch 关联及候选字段的跨形状自洽。不能用后来的 ByEntry 追补早先 name-only launch。
4. 两 rank 报 selected=written=7296、pending=0、无 drop/error、逐位金丝雀一致；实际完整性还要在黄区对封存文件跑 `review_launch_capture_v2.py`。任何缺文件、截断、计数不符都退补。函数名哈希记录可作为原始 host 数据保留；没有真实 key 时单列“不可判定”。

独立 r2 已覆盖相同的按名加载需求，但采用显式 entry/name_only 类型，不复制高位混用：只按名加载、后续提升、entry 优先、失败 lookup、9000 条完整捕获等 CPU 用例已有记录。两份源码不同是有意设计，不声称逐字节相同。

## serializer 边界

锁定公开候选 `v9.1.0 / c77c0829f1f0f0b8f882db1d9b051a758b605472` 的 `FAInferTilingData` 探针包含 `coreInfo` / `splitInfo`（coreNode / splitNode），2488 字节。**当前 schema 不含 `outerSplitParams`、`fdParams`**。黄区这两个名称的来源在原文中也为 UNKNOWN，不能仅凭名字把它们重命名到 coreInfo/splitInfo，或凭 V1 推导布局。

需补该真实结构的声明路径、版本/文件 SHA、顶层类型、成员次序/类型/数组长度，以及与目标 key、硬件分支的对应。若 N2 显示实际为此其他结构，应新增独立 schema 和本机 SDK 探针，而不是继续套用 2488 字节。当前只保留已有候选解码，拒绝不匹配的结构；没有材料时不猜实现。

## 下一步

先按 [N2 离线执行单](fault33_n2_existing_capture_20261124.md) 提取六条目标记录。缺少的字段保留 null，写清能否从旧 trace 补到。只有旧 trace 没有建立唯一关联或没有必要 host 数据时，才针对缺项另开采集窗口；不是统一重跑全部身份/金丝雀。

收到 N2 后锁定 serializer / key / 分组，才给窗口二的最终参数。五条标准仍为：公开候选身份、字段边界、三组跨形状、每组固定真实 Q/K/V 全输出逐位一致率 ≥99% 并定位差异、任一不满足即停。A2 参数不能替代 A3。33 当前保持关闭。
