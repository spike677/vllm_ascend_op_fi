# Fault26 B/F1024 msprobe：8-step 配对采集

本目录新增本次 B/F 数据，不替换历史 Fault26 1024/4096/nohit。
注入逻辑见 [INJECTION_LOGIC.md](INJECTION_LOGIC.md)。

## 配置及文件边界

- `task=statistics`、`statistics.summary_mode=statistics`、`level=mix`。
- step0–7，rank0/1；同步采集。step0 是7-token prefill，step1–7 是 decode，不是8个输入token。
- 每包50个 JSON：16份 dump.json、16份 stack.json、16份 construct.json、采集时开启的 dump_config.json、probe_config_ack.json。
- 每个 dump 含2133条记录；JSON、ZIP CRC、step/rank覆盖、压缩前后字节一致性均验证通过。SHA256_MANIFEST.json 包含包及每个成员的hash。
- 只提交统计dump与说明。不含完整tensor、response文件、native trace、服务日志、源码、二进制、失败尝试或R组。

| ZIP | 压缩大小(bytes) |
| --- | ---: |
| Fault26_Baseline_B_ON.zip | 3360548 |
| Fault26_Fault_F1024_ON.zip | 3361921 |

## 配对与输出

模型：Qwen3-32B-w8a8-nopdmix；A2/910B4、CANN8.5.1、TP2/eager。
vLLM-Ascend commit：`99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f`。

两组独立启动，request JSON和实际prompt token IDs一致：

```json
{"prompt":"What is the capital of France?","temperature":0,"seed":0,"max_tokens":8,"ignore_eos":true,"logprobs":5,"return_token_ids":true}
```

实际prompt IDs：`[3838,374,279,6722,315,9625,30]`。

完整输出（JSON字符串保留空格）：

```text
B: " The capital of France is Paris. Paris"
F: " The1    , "
```

```text
B token IDs: [576,6722,315,9625,374,12095,13,12095]
F token IDs: [576,16,220,220,220,220,11,220]
```

均为8个生成token、finish_reason=length；故障文本在第2个生成token分叉。为固定采集8步而设置ignore_eos，不可据此判断自然停止或长文本表现。

## 注入与证据限制

- B：0 applied。F：预算1024，实际447 applied，447 restored；1024不是实际hit数量。
- 441次具备consumer完成及恢复证据；其余6次 restored 记录为consumer_matched=false、consumer_executed=false、pass=false、real_execute_status=1，原因尚未定位，不能将447次全部标记完整闭环PASS。
- 447次恢复hash均等于原值，restore/readback状态成功。没有额外新请求recovery，不能宣称新请求恢复验证通过。
- 控制面已disarm、观察wrapper已卸载、两个实验容器已停止，结束时NPU2/5无运行进程。
- 原生inventory每组200000行，达到配置的每worker100000事件上限；不可把它当作全程无遗漏trace。
- 源runner的PAIR_PASS验证采集/计数，不代表加强审计中的全部consumer链路通过。

上述runtime结果来自保留在本地/远端的完整证据，不是这些dump-only ZIP自身能够独立证明的内容。

## 首次差异：不是从embedding开始

- step0：两rank所有同名统计记录相同。
- step1：两rank embedding仍相同；rank0首个可见差异为 `_C_ascend.npu_add_rms_norm_bias.2.forward` 的第二输入（residual），其Norm从28.375变为8.25，第一个输入统计不变。
- rank1 step1首个可见差异为 `Distributed.all_reduce.4.forward`。
- step2起，先前生成token已分叉，embedding输入随之改变；这不能作为embedding故障证据。

统计相同不证明tensor逐元素相同，首个dump差异也不等于最底层首次非法写入。

## 保留的完整证据

远端：`/home/haoran/fault26_msprobe_bf8_20260916_v2`。

Windows：`C:\Users\admin\Documents\demoncatpfi\fault26_msprobe_bf8_20260916\records_v2`。

完整包：`B_F1024_records.tar.gz`，15272588 bytes，SHA256：
`4d9c4a475633276a694d98fa74c25e9196bdeba0912261020eb84b8af420f52b`。

完整包之外的本轮审计报告：本地 `fault26_msprobe_bf8_20260916/README.md`，远端 `POST_COLLECTION_AUDIT.md`。它们明确补充6条consumer证据缺口。
