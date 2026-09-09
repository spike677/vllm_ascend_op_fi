# 仓库分层与渐进迁移门禁

## 决策

canonical repository 为 `D:\project\demoncatpfi`。`C:\Users\admin\Documents\demoncatpfi` 是只读 legacy/investigation clone，不作为开发或文档依赖入口。

本仓库是 Ascend/NPU backend 研发与验证来源；隔离的 DemonCAT 项目继续拥有产品框架和生产控制面。项目正从按 Fault 编号堆叠的探索结构，渐进迁移到可复用的 backend/strategy/observer/evaluation 结构，但不会在没有验证的情况下批量搬动 native runtime。

```text
src/ascend_fi/         可复用实现和集成边界
profiles/capabilities/ 可复用 capability metadata
experiments/           历史案例、配置和冻结策略
fault20_fast_validation/ 已验证 legacy runtime/source/evidence tree
docs/                  文档入口与迁移解释
```

## 职责分配

```text
src/ascend_fi
  = backend、strategy、observer、discovery、validation、evaluation

experiments/faultXX
  = 历史实验参数、配置、证据和冻结说明

profiles/capabilities
  = execution plane、支持范围、验证条件和限制

DemonCAT 产品
  = PrimitiveSpec、RuntimeBinding、controller、授权与生产 CLI/API
```

物理字段（调用序号、device address、tensor slot、stream、allocation generation、cache key）仅属于某次 discovery/binding 事实，不能因为一次实验成功就提升为稳定用户 ABI。

## 单组件迁移门禁

1. 在 capability manifest 记录现有 canonical source 与 evidence；
2. 分清 backend、strategy、observer、lifecycle 职责；
3. 仅迁移一个经过验证的 source of truth 到 `src/ascend_fi/`；
4. 保持历史 experiment 的兼容构建路径；
5. 重新通过 ABI、disabled baseline、fault 与 recovery E2E；
6. 更新 manifest 指向新 canonical implementation；
7. 最后才将历史目录收敛为配置、hash、evidence 与 pointer。

任何未通过这些门禁的能力不得因为文档归位而被升级为产品 capability。
