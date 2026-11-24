# 33 r2补充件交付（2026-11-24）

N1—N4本轮蓝区动作已完成，A3窗口一仍待回传。r7.4.1标签和产品包不变；main已按顺序快进至r7.4、r7.4.1（59f6aea5）。本补充件在独立分支交付，不把工具修复伪装成新的产品资格。

- ZIP：`fault33-evidence-candidate-20261124-r2/fault33-evidence-candidate-r2.zip`
- SHA256：`962361e8cc3f2205c30b1c57be8d8aa561b9e75be0cc92cb432c46a0c3e4ff11`，106270字节。
- 内容32项：候选工具、修复采集源码、Worker辅助文件、构建器、参考aarch64库、构建和测试回执、两窗口执行单及接收清单。外层SHA256SUMS同时覆盖manifest、README、build-receipt与package-check。
- 36项Python CPU通过，9项模拟ACL通过，CANN9.1无设备编译通过；实际r7.4.1源码＋r2 ZIP解压组合再次36/36通过。r2补上旧包缺失的fault33_capture_worker.py。
- 新源码SHA：2d472c2e3268080c52bfa3ba497e6239a22f77e062849bbed379e5177578a835。
- 参考库SHA：c87db1e157bca28e395e1c8280b0958fdea2d057b29affa21576c727996e023e；A3需按实际SDK重建和核对映射，不以A2参考库直接登记。

先读`docs/worker_core_v1/yellow_intake_20261124_fault33_full.md`，再按两窗口执行单的r2勘误运行。仅按名字加载的记录可以完整采集，但真实key仍为null，不能进入解码；ByEntry后到只影响后续记录。GDN/unified_attention只计数，目标是16层full attention中的真实FIA decode。

黄区原有9库身份和逐位金丝雀可复用；单形状和自改工具结果不作最终窗口一依据。须三形状×两rank唯一关联，补launch key、blockDim、tiling边界、mask、实际KV长度及block_table范围。窗口一审查后才配置控制重算，33保持关闭。

Qwen线统一正式测试用r7.4.1，范围见`RC3-PREVIEW-R7.4.1-QWEN-FINAL-ADDENDUM-20261124.md`，不再单独安排r7.3.1正式测试。本轮无NPU占卡、无新抢占隔离，两个无设备构建容器均已退出。由负责人转交，不直接联系黄区。
