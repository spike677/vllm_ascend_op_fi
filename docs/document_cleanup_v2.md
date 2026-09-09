# 文档清理 V2

## 结论

V1 inventory 的 **237 份 Markdown、66 个 SHA 组、129 个额外相同副本**是四个来源合并的盘点，不是同一个 Git tree 中的重复文件数。相同文件在 canonical、evaluation、research worktree 与 legacy clone 各有一份，是分支镜像，不能为了“去重”删除其他 worktree 的文件。

本轮以现有 inventory/migration manifest 为起点，复核当前集成分支 tracked Markdown。当前没有同一 Git tree 内 SHA 完全一致的重复 Markdown，因此**删除 0、物理归档 0、新增 redirect 0**。四份常被提到的 Fault20 报告各有不同 SHA；canonical docs 页也是提炼后的不同内容，不能以“内容大致相同”替代 exact-SHA 删除门禁。

V1 的 `DUPLICATE_STUB` 是跨来源盘点建议，不是执行授权；本轮在 JSON 审计中逐项标为跨来源只读保留。已有原位历史报告、archive catalog、evidence pointers 和旧链接均保留。

## 本轮变更

- 新增 Distortion V1 中文说明与入口；保留核心 metrics 和完整 Analysis design。
- 为跨来源 duplicate 与当前树的真正 duplicate 分开计数。
- 更新 migration 的当前 hash，同时保存 V1 hash。
- 重新检查 canonical `docs/` 的相对链接，结果见 [link validation](document_link_validation_v1.json)。
- 清理明细见 [V2 audit JSON](document_cleanup_v2.json)，原始 [V1 inventory](document_inventory_v1.json) 未覆盖。

不改 C 盘 legacy clone；不改真实 evaluation/research 分支；不删除或搬迁 raw evidence、JSONL、msprobe dump、bundle、runtime。原始 `main` 的两处未提交 Fault26 修改保持 `PREEXISTING_DIRTY_UNTOUCHED`。
