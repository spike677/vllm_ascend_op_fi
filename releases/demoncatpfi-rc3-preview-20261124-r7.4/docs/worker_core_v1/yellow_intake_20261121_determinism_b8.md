# 黄区文本接收：同进程确定性与首批B8

来源：openchamber文本，由负责人归档至 `D:/CodexArchives/20261121/yellow-dsv41-determinism-qwen-tp2-b8`。原文4,163字节，SHA256 `a43aaf39e1ae8f9e18346f0e59771f9785a55e507611f4e5719b86c231f12cb5`；已核对归档manifest。原件Git外，本文不记录机器/设备编号或内部路径。回传文档提交c7a355e。

## 观测与范围

- Qwen3.8 W8A8、A3、0.26rc1-a3、TP2、B8：新进程首批8条相同5-token提示，生成2token；三轮24请求均为`[11751,13]`。第2请求top-5输出logprobs与prompt logprobs无NaN/None，首token top-5三轮相同。**没有采集完整hidden/logits**，不能据此证明整个张量有限或逐位一致。
- A3这次检查未出现A2观测到的首批GDN异常。提示、软件栈/驱动、硬件和采集粒度不同，不能宣称A3普遍无该缺陷；本记录仅为黄区文本证据。
- DeepSeek V4.1 W4A8、TP8+EP、32K严格实例：同进程三次固定请求，temperature0，生成8token；共同前缀仅3token，第一/二次第8token分叉，第三次第4token分叉。同进程基线不稳定，不能只归为进程重启差异。

## 对验收的影响

Qwen仍保留最多6次预热、连续两次完整hidden/logits全有限且精确一致的门禁；logprobs局部检查不能替代该门禁。蓝区B8通过不迁移为A3准入。

DeepSeek公共入口精确恢复验收前，需等黄区确定性排查结果（含实际HCCL_DETERMINISTIC等设置）。不把容差噪声方案自动启用，不登记V4.1组合，不安排蓝区V4.1 NPU。

32K统一为chunked prefill开启；本次5-token请求小于4096批token预算，可以提出单chunk验证，仍需实际调度证据。此前关闭chunked的Qwen测试只记录为历史配置，不用于证明新模板行为。

缺失字段：确定性环境变量实际值、完整hidden/logits、完整软件栈与源码指纹、实际调度chunk映射。仅供环境缺陷定性及RC3前提分析；`product_closed=false`，没有据此作新组合登记。
