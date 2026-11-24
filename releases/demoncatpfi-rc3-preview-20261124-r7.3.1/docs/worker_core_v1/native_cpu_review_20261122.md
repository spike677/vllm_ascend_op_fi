# Native 公共入口与 MC2 准备：CPU 回归

2026-11-22，扩展 RC3 子集共 99 个测试模块、1023 项测试，0 失败、0 错误、2 项已废弃研究入口跳过，耗时 185.703 秒。完整模块、跳过原因见 [结果清单](native_cpu_review_20261122.json)。这不是全部历史测试欠账清零，也不是 NPU 验收。

覆盖 Native 对象区间身份、同 stream、恢复保护，41/26/20 site、公共 Runtime、准入、独立回执审查、发行检查，以及 MC2 原版本方法签名和 CPU 对象关联。20 的 1216 次调用证据逐项检查后按实际站点汇总；测试确认不提高 512 KiB 上限，也不丢弃失败恢复记录。接收端 expert 行映射仍未完成，CPU MC2 通过不授予 DeepSeek 资格。

首轮 1020 项中一个 subprocess 测试因测试启动器只设置 `sys.path`、没有向子进程传递 `PYTHONPATH` 而导入失败；修正启动器环境后，原失败用例及相关 39 项通过，再运行本轮完整子集。未改生产逻辑掩盖导入失败。失败原始记录仍保留在本地私有暂存。

复跑需把仓库的 `src`、根目录、`tests`、`tests/worker_extension`、`fault39/src` 同时加入父进程和子进程的 `PYTHONPATH`，再按 JSON 的 `modules` 列表用 unittest 加载。两个跳过均为已经关闭的旧研究公共入口；替代的普通准入拒绝测试仍执行。

另外，CANN 9.1 SDK 构建执行了三个 ACL 替身 C++ 单元测试；其身份和结果见 [Native 构建回执](../../experiments/worker_core_v11_20261118/native_regions_build_20261122.json)。编译通过与 A2 实测、A3 待构建三种结论分别记录。
