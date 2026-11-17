# 分支与 PR 规范

历史基线 `codex/17-fault-c2c3-closure-20261117` / `9dfdb0c` 保持原样。统一工程只在 `integration/demoncat-unified-api-v1` 集成，不把历史实验分支整体合并。

公共能力使用 feat/fault-registry、feat/unified-config、feat/operator-contract、feat/trigger-engine、feat/kernel-correlation、feat/fa-fia-entry 等主题分支。接口、配置、注册表相互依赖时 PR 描述必须列出依赖；逐个合入工程分支。每个可集成 PR 必须通过 Adapter Conformance Suite、生成文件一致性和受影响的旧接口测试。Native/Worker 资格单独审查。

API、配置、证据 schema 均显式版本化。破坏性变化新增版本并提供迁移，不修改旧实验记录；正式版本标签在实际发布评审后创建。当前版本不创建 NPU 已验收的发布标签。

PR 描述列出具体行为变化、有效执行机制、NOT_IMPLEMENTED 范围、回归结果和环境限制。统一注册数量不得替代实机验收数量。
