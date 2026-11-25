# 33 N2 六条记录摘要接收：部分字段已补，窗口二仍不放行

来源：负责人提供的黄区纯文本《现役采集记录_N2六条目标记录与采集缺口》，本地接收目录 `D:/CodexArchives/20261125/yellow-fault33-n2-six-records/`。原始 JSON、capture 和完整回执仍在黄区；本次没有收到 N2-SIX-RECORDS.json 全文或规范提取文本。不转载主机地址、硬件编号、容器名或进程号。

接收原文：3208字节，SHA256 `071cbeb8301441673f694a9a1a3a46a028950102ecdec8eb0261306ca27e52fc`。

## 已补内容

三组 prompt（128、512、2304）×rank0/1，摘要均给出：prefill Q=[L,12,256]、K/V=[L,2,256]；首个 decode KV 长度分别129/513/2305。共同 decode Q=[1,12,256] BF16、stride=[3072,256,1]；K/V=[8532,128,512] BF16、stride=[65536,512,1]；mask INT8 [2048,2048]、stride=[2048,1]；TND、block_size128、sparse_mode3、Q头12/KV头2、scale0.0625。每形状每rank记录16 prefill和48 decode。

这补齐了 API 形状摘要，可准备控制重算输入规格；不等于六条 launch/tiling 证据已经闭合。完整度材料还报告一个 rank 的 census seen/written 为0，且窗口与进程的对应是按运行顺序推定，不能作为已验证关联。其余五条也须先证明窗口关联，才能接受相应完整度结论。

## 审查结论：需退补

11-25复核澄清：黄区已生成六条N2 API记录，不再记为“尚未采集”。蓝区收到的是原文摘要和真实源结构 `shapes[0..2].attention`，未收到六条launch对应的规范提取文本。当前待办是从已有原件补提/核对，而非重复采集六条API。负责人补充的三点（真实键pointer映射、80字节header与未知项null、r2-k3工具来源）已写入 `fault33_n2_existing_capture_20261124.md`；原文没有这些字段不算采集内容已具备，不以手工默认值补齐。

1. 六条各选同一 full attention 模块的一个 Q1 decode；回传原 API 调用序号、模块名、rank、有效 block_table 页面范围、实际序列边界。不能只有各形状的总调用数。
2. 按 `fault33_n2_existing_capture_20261124.md` 的工具从既有原件提取：capture header 原始 key/blockDim/serial、host 字节数/SHA、API→launch 唯一关联及证据来源。窗口与rank映射必须来自日志关联，不按运行先后猜测。
3. 公开候选 serializer、sparseMode3 分支、实际加载库、SDK布局及相关文件需要完整 SHA。当前 `attention_v1.py` 和 manifest 仅给 SHA 前缀；不能用于严格身份核对。tiling区间未审核前继续为null，不套A2偏移或长度。
4. 回传 N2六条提取文本及原件/映射/提取文件名、字节数、完整SHA。原始文件留黄区。先核对已有 capture 与 HTTP/trace，不要求全部重跑；只有确实没有记录的那一窗口/rank才考虑补采。

## 五条门禁现状

|门禁|本次材料能支持的结论|
|---|---|
|公开源码候选身份|仍需与实际SDK/库及分支绑定，未闭合|
|字段边界校验|API字段部分齐；host/tiling边界与placeholder来源未齐|
|三组形状对照|有三形状API摘要；无三形状逐条launch解码对照|
|每组控制重算逐位一致率≥99%并定位不一致|尚未执行/未收到，不能宣称通过|
|任一不满足即停止放行|保持33 A3关闭，不冻结r7.5资格|

负责人转交上述缺项即可；蓝区不直接要求黄区执行。此次材料不改变32/34的A2结果，也不成为任何蓝区实机证据。
