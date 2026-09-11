# 产物与证据规则

源码发布清单不是完整原始证据包。Git普通交付包含源码、文档、冻结公共样本、必要测试夹具、哈希索引；完整request/response、pair/raw、部署配置、服务日志、trace和大dump留私有包。密钥、access.token、认证头绝不提交。

派生中文报告、统计、Dashboard生成到新目录，记录输入/评分源码hash，不覆盖旧文件后沿用旧hash。

## Qwen0.6B已有证据

原始包仍在产生它的历史工作树：
`D:\project\demoncatpfi-worktrees\interactive-inspect-v1\artifacts\inspect\private\qwen06b_demo_20260911`。

这是证据位置，不是canonical源码运行依赖。远端证据根 `/home/gyl/demoncat_demo_qwen06b_20260911` 本轮不启动、不改写。包中的 `evidence_v1/` 是Inspect/计划/IDENTITY/pair/raw/收尾，`user_report_v1/`是已有修正统计与连续原文Dashboard。

发布[哈希索引](../docs/releases/source_baseline_v1/evidence_index.json)用于取得私有包后核验。ready/stopped只证明当时状态，不是实时服务器状态。

其他 `fault20_fast_validation/reports/`、`evidence/`、`msprobe_dump/` 原位置/原内容保留，本轮不迁移、压缩或清空。
