# 33 A3 TP2：限时补证执行单

仅取证，公共入口继续关闭。使用Qwen3.8/A3/0.26/CANN9.1真实组合；不要换9.2库。
本单由负责人转交，不直接运行黄区。建议最多两个30分钟只读窗口；失败时提交缺项与结案建议，不自动延长或重跑注入。

## 1. 来源、二进制、serializer

在实际Worker容器中记录已加载operator/host库的真实路径、SHA、ELF build-id、CANN runtime版本、torch_npu/vllm-ascend源码SHA及安装包来源。
只查源码/元数据，可运行以下命令（实际目录由现役环境给出，不据目录名推断版本）：

```sh
find /usr/local/Ascend -type f \( -iname '*incre*attention*' -o -iname '*fused*infer*attention*' -o -iname '*tiling*data*.h' \) -print
```

从当前operator包提供者的匹配source包/源码提交取得该分支serializer/tiling struct及注册key代码；若本机缺文件，可由部署方取得公开或合法内部source包，记录URL/发行身份/commit及逐文件SHA。
源码与已加载二进制必须有构建清单或可复核的包来源对应关系。仅“都叫CANN9.1”不够；没有对应依据即serializer未证明。
公开源码检索入口：[ops-transformer 的 FIA V3 接口文档](https://github.com/hicann/ops-transformer/blob/master/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV3.md)。2026-11-24可读取；这是源码定位入口，**不是已确认匹配黄区二进制的版本**。文档把sparseMode作为API参数，并单独规定mask/序列约束，不能从它推出唯一tiling。
取得候选源码后执行 `git remote get-url origin`、`git rev-parse HEAD`、`git status --porcelain`，并用 `rg -n 'sparseMode|GenTilingKey|SaveToBuffer|BEGIN_TILING_DATA_DEF' attention/fused_infer_attention_score` 找到实际定义；逐文件SHA与包构建身份核对失败即停止。AtomGit CANN入口本轮浏览不可读，不宣称已下载匹配源码。
只用serializer声明的字段类型/偏移/大小解码raw；公共POD单独保存。不能用A2偏移、猜测字段或期望profile解A3 raw。

## 2. sparseMode=3与真实key

先记录API sparseMode、mask形状与值语义、actual sequence lengths、block table和目标query；再沿匹配host源码的条件分支找到实际tiling key注册、模板、scheMode与S2分组来源。
sparseMode不是scheMode，数字3不能直接翻译成某个核内分组；同key仍需核对实际raw/POD参数。
输出证据表：API参数→host条件与源行/SHA→实际launch key/blockDim→serializer字段→模板/合法key集合。任何一环缺失保留null。
对同一真实Q/K/V控制重算时，必须用该sparse规则的同一合法key集合。不能把全causal集合强套到稀疏路径。

## 3. 7928条与serial8191

当前公开采集器paged33_launch_capture.cpp的两处上限为entries.size()<8192和serial<8192。
serial在生成文件前就分配；随后args空、size>65536、count>128、创建/写文件失败均可能使serial存在而没有完整文件。
因此7928与8191不能证明只有264次未执行，也不能直接归咎于JSON截断。
按pid/rank分开统计文件数、header serial最小/最大、唯一数、重复、缺口、实际文件长度与header长度相符情况；最大8191只说明触达该上限的候选，不证明此前没有其他过滤。

取得完整证据的顺序：
1. 先保留现有二进制原件，逐文件SHA和清单总SHA。不要从已截断stdout重建raw。
2. 已知实际FIA key后，使用经过审核的新只读采集器：按精确key过滤；由宿主在预热后arm、单请求完成后同步并disarm；序号按进程/窗口单调，不能静默回绕。
3. 新采集器须记录seen、selected、written、filtered、oversize、open_error、short_write、capacity_drop及start/stop，保留所有写失败；写全、flush/close检查成功后才计written。
4. 依据本次实际调用量设定有界容量和磁盘预算；达上限必须明确capture_incomplete，不能把“增大8192”当完整性证明。独立新进程/短窗口可降低量，但不能漏掉目标prefill/decode。
5. 原件只留黄区。文本分段≤4KB，给段号/总段数/本段SHA及原件总SHA，传字段摘要，不上传完整QKV。

旧采集器没有上述计数和arm边界，本单不宣称它能通过环境变量开启不存在的功能。若黄区无法提供经过审核的完整采集器或匹配serializer，停止此窗口，报告缺项；不靠反复跑相同截断脚本。

## 4. 判定

完整捕获+serializer对应+实际分组明确后，固定真实Q/K/V、mask、序列边界和页映射，比较完整目标输出逐位一致率≥99%。不得挑选局部元素或改变阈值。
通过再做同Worker两个owner的产品闭环：全rank真实消费者、模型输出变化、区域/资源/金丝雀恢复；不要求token变化。
任一证据不能闭合则建议结案为“A3当前分组路径尚未取得可审查的控制重算证据，保持关闭”；只有证明本部署分支不满足适配条件时才建议“当前部署不适用”。
结案由负责人决定，不把缺源码/截断误写成算子不支持，不撤销A2历史通过。
