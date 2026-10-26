# msprobe 原始值证据

124 / CANN 9.1.0 / Qwen3-0.6B，mindstudio-probe 26.1.0，`tensor/debug` 采集。仅交付原生msprobe数据和采集配置，附本索引与校验清单。

| 包 | 对照含义 |
|---|---|
| `B_msprobe.zip` | 正常推理 |
| `I_msprobe.zip` | 同值权重重建控制 |
| `F_msprobe.zip` | 首个cold prefill一次错误逻辑权重读取代理；warm不注入 |
| `R_msprobe.zip` | 独立恢复组 |

每包54个原始文件：52份BF16 `.pt`、1份 `debug.json`、1份采集配置。合计208份张量、4份debug JSON、4份配置；包含输入、目标层权重快照、真实linear输出及logits，均按源字节保留。`SHA256.json` 列出发布文件和ZIP成员的哈希。

四包可解压至同一空目录，分别得到 `B/`、`I/`、`F/`、`R/`。在各组的 `native_msprobe/step0/rank0/debug.json` 中，`data_name` 对应旁边 `dump_tensor_data/` 下的张量文件。配置和JSON中的绝对路径保留服务器原始值，不代表本机必须存在该路径。

`step0` 是一个debug采集会话；文件名 `cold_s00` 至 `cold_s07`、`warm_s00` 至 `warm_s07` 分别对应两次请求各8次真实模型执行。每组保存48份输入/输出/logits及4份权重快照。该格式提供逻辑张量值，与此前的statistics摘要不同，也不保留NPU物理私有布局。

F为已获准的错误视图读取代理，不代表历史“私有格式as_strided原地写回失效”缺陷已复现。
