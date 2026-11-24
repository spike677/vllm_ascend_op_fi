# 故障33：黄区完整报告接收与退补清单

来源：负责人转交《故障33_A3取证完整报告_原文.txt》，档案批次 `yellow-fault33-a3-full-report-f279a79a`。原文留仓库外；本页不复制机器地址、容器名、私有路径和硬件编号。组合为黄区 A3 / Qwen3.8 W8A8 / TP2 / v0.26.0rc1-a3 / CANN9.1.0。证据类型为黄区文本声明，不是蓝区实测。原文 SHA 和字节数见同名接收 JSON。

**结论：部分材料可沿用；仍需退补，不能进入控制重算或启用33。** 报告的“步骤1–5完成”不等于新版三形状窗口一通过。

## 可以沿用

- 9个已提供的库身份，见 `yellow_intake_20261124_fault33_library.md`；本轮只核对使用环境是否变化，不要求重新计算全部SHA。版本文件是安装元数据，不能重标为运行时API读数。
- 完整性审查方法：按rank核对封存、序号、文件数及所有错误计数。报告称每rank 7296条、unknown=0、pending=0；这是待替换工具版本下的结果，不能直接作为新版窗口的最终证据。
- 金丝雀文本结果：每rank四个生成位置，BF16 `[1,248320]`，每位置全部元素逐位一致、非有限值0。保留原结论，不单独要求重跑；如新窗口部署或输入发生变化，应明确变化，不能跨组合借用。

## 不能沿用为最终取证

1. 仅2304-token一组形状，不能代替同窗口串行128/512/2304三组真实输入。
2. 黄区自改采集器7203字节，源码SHA `1fecdc87bb4ea3f923d25d9d57e4bad83da41060d7c986bdc1c2635b0b24a4dc`，库SHA `93df16e8d734c5ddb55a3e78fe8f46f1c03914defeb8a3ec39e9ec3f4627e15e`。未收到源码原件，不声称已审查或逐字节合入。此次回传用于确认按名加载漏记问题；正式窗口改用蓝区独立实现的r2。
3. `unknown=0`只说明句柄可归属，不证明每条记录都有真实entry/tiling key。r2将name-only记录的key明确表示为null；不能用函数名、临时0或哈希推导真实key。
4. API共320次、三类算子计数，尚未唯一关联到7296条launch中的目标记录。FIA、GDN core、unified_attention包装层不能混为一个目标集合。

## 下一次仅补窗口一

用独立r2目录及其构建回执；在A3实际SDK核对头文件、布局、C++ ABI和52字段marker roundtrip。一个TP2实例中串行128、512、2304-token输入，各生成4 token；各rank分别报告：

|字段|要求|
|---|---|
|工具身份|r2 ZIP/SHA、源码SHA、实际加载库SHA、编译器和SDK身份、原构建回执SHA|
|输入/请求|输入token数及SHA、真实内部request ID、prefill/decode阶段及生成位置|
|API归属|API调用号、实际full-attention层/模块、调用栈或唯一关联依据；不能按最近序号猜配|
|目标launch|记录文件名、rank、pid、serial、bin SHA及binding sidecar SHA；真实entry/key及key_source|
|launch原值|blockDim、host args字节数、placeholder数量/宽度及已核对的holder索引|
|tiling|实际tiling字节数与边界；字段从原件解码获得，候选2488不是可预填的实测值|
|注意力元数据|Q/K/V实际shape、dtype、layout、Q/KV head数、head dim、mask存在性/形状/语义、sparseMode|
|长度和缓存|实际query/KV长度、block_size、block_table有效元素范围和page ID边界、尾块有效长度|
|完整性|每rank seen/selected/written/unknown/filtered/所有drop/error、封存/pending；name-only记录单独计数|
|非目标|GDN core及unified_attention仅计数/说明包装关系，不能当作FIA解码结果|

每组三形状均需找到同一full-attention模块的真实decode FIA；三组×两rank分别列目标记录。仅有名称匹配但entry未知、多个launch无法唯一关联、mask语义不明、真实长度缺失，都保持null并停止该调用的解码。不能直接用128/512/2304填KV长度。

完整原件保留黄区，回传紧凑文本、上述字段及原件SHA；一段放不下时按形状/rank拆段，不截断关键字段。9库身份引用前批记录。补充由负责人转交，Codex不直接要求黄区运行。

## 窗口二与停止条件

窗口一审查通过后才给出具体分区/算术配置，不能自行套用A2的512或在多种算法中挑最高一致率。固定真实Q/K/V控制重算仍要求：公开候选身份、字段边界、至少三组真实形状、每组每rank完整输出逐位一致率≥99%及全部差异位置。任一不满足就停止并说明原因。两owner注入验收是之后的独立阶段；本次接收不生成33登记。
