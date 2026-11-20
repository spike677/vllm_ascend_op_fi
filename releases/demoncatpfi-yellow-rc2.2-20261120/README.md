# 测试部署线 RC2.2

源码 `ec941ed6d2a166443d208e1bb9a3284667ae4cb1`，基于 `d07c7215`；标签 `demoncatpfi-yellow-rc2.2-20261120`。旧冻结标签和包不变。此版冻结后测试部署线仅维护，多卡能力属于RC3。

三包版本：demoncatpfi 0.3.2、worker-extension 0.2.2、fault39-core 0.2.2。发行源码从842个固定Git blob导出；290个已安装Python源文件与源包逐字节相同。124 aarch64无设备、无网络runc安装通过，未启动Worker、未构建Native、NPU用时0。候选137项CPU回归通过（1项跳过）。

## 范围

- 41普通任务：normal→normal_repeat→lossless→fault→off→health；蓝区A2/CANN8.5.1/vLLM-Ascend0.18/Qwen3-0.6B BF16/TP1、同Worker两个作用域。证据文件a3是attempt，不是A3硬件。
- request_plan：查询和任务创建共用阶段定义；普通四请求任务、41六请求、Batch六个B8批加前后健康。宿主enqueue→advance/advance_batch→等真实完成→query读取next_phase，直到complete；之后还需清理、核对资源为零。
- prerequisites缺项汇总：pcp_size缺失显式报告，不填默认值、不放开新栈。
- 18/39/Batch：收录蓝区A2/0.18/Qwen3-0.6B TP1同Worker两作用域的既有补验。18/39为BF16；Batch为W8A8、B8。Batch的完整hidden/logits health补验与早期token-only证据分别保留。
- 22总结工具保留Q/K前后完整身份、correct/replacement rotary行、独立reader、恢复字段。旧回执未记录的字段仍缺失，不补造；总结上限64KiB。
- 28仅保留显式研究入口，公共入口仍拒绝；20、26、33、15、25及39 matmul_range不因本发行成为普通可执行能力。
- 继承RC2.1的18/39/22精确黄区测试组合登记，证据为YELLOW_TEXT_ATTESTED、原件留黄区。A3/Ascend910_9382/CANN运行8.5.1（toolkit文件8.5.0）/Qwen3-0.6B BF16/TP1；18黄区两作用域不在同一Worker。22行级效果未进入旧summary，以hidden/logits变化及OFF/health恢复为证；本版修复总结器，但不改写旧证据。
- 8、17只沿用蓝区已验收范围；Batch黄区待配置；41黄区未登记。未验收组合默认关闭。不包含TP2代码、不代表A3现役或Qwen3.8验收。

## 交付与核验

[源码包](demoncatpfi-yellow-rc2.2-20261120.tar.gz)、[源码清单](source-review.json)、[安装核验](install-summary.json)、[发行清单](release-manifest.json)、[校验清单](SHA256SUMS)。

许可中的 `source_manifest_sha256` 使用 **source-review.json文件本身的SHA256**：`1a9e2c5cf244c6f2abaebec877b527099d1805faf2aed7f6bd3622da1d66f463`，不是release-manifest.json。
源码包SHA256：`b3b6af57df8a787d78dd7df5ef22035070737d75d975c900269437034dda902e`。部署组合/许可仍按公开准入契约校验，发行版本不自动授准。

新空target安装三个SHA核验wheel，执行源码包内tools/check_rc2_install.py；按install-summary逐文件核对已安装源码。可信部署包装入口为fault_injection.rc22_profile.prepare_binding；原rc2_profile不变。原生provider必须按实际部署重新绑定，不能把蓝区Native制品搬作黄区许可。

本版未发送黄区；master升级仍按已交付RC2.1步骤，由负责人决定。产品闭环以黄区实际验收为准。
