# 唯一命名规范

四层身份：Fault ID（F008…F041）、Mechanism ID（如 F028.FIA.DENOMINATOR_REPLAY）、Fault Family、Injection Primitive。没有历史数字编号的 DynamicQuant 使用 FQUANT，不虚构新编号。17 项机制可以属于同一 Fault ID。

legacy_id 保留 `8.restore_offset`、`28.fia.denominator_replay` 等现有拼写；不改历史文件、CLI 参数或原始证据里的名称。canonical_id 是稳定新别名，两者一对一。故障家族不是执行资格，Primitive 描述实际动作而非根因已复现。

唯一人工维护机制条目为 registry.yaml。新增机制时定义语义、专属参数 Schema、Adapter 能力和兼容声明，然后生成目录、卡片、矩阵、API 索引与任务队列。生成文件禁止手工改写；CI 执行 generate --check。旧固定验收快照保留其提交范围。
