# 文档迁移映射 V1

> 本表记录本轮 canonical 文档入口如何从既有 source/report 归位。它不表示原 source 被删除；evidence-linked report 保持原路径。

| 旧路径 | 来源 | 新 canonical 路径 | Source SHA-256（前 12 位） | Action | 原因 |
| --- | --- | --- | --- | --- | --- |
| `docs/architecture/repository_layers.md` | `canonical` | [`docs/overview/repository_layers.md`](overview/repository_layers.md) | `2669a1127842` | `MOVE` | 旧路径保留 redirect notice；中文 overview 为唯一正式维护入口。 |
| `docs/integration/demoncat_merge_contract.md` | `canonical` | [`docs/overview/demoncat_merge_contract.md`](overview/demoncat_merge_contract.md) | `1016a53b4c82` | `MOVE` | 旧路径保留 redirect notice；中文 overview 为唯一正式维护入口。 |
| `README.md` | `canonical` | [`docs/overview/project_overview.md`](overview/project_overview.md) | `f3205dc70ddb` | `COPY_FROM_BRANCH` | 从 canonical README 提炼仓库边界，不搬动 root README。 |
| `README.md` | `canonical` | [`docs/overview/terminology.md`](overview/terminology.md) | `f3205dc70ddb` | `KEEP_CANONICAL` | 新增术语页，避免模型/故障报告各自定义术语。 |
| `docs/evaluation/fault_effect_evaluation_design_v1.md` | `evaluation_worktree` | [`docs/evaluation/fault_effect_evaluation_design_v1.md`](evaluation/fault_effect_evaluation_design_v1.md) | `c4823665fb1c` | `COPY_FROM_BRANCH` | 保留工程设计文档，改为中文 canonical 版本。 |
| `docs/evaluation/qwen_2card_fake_requant_current_effect.md` | `evaluation_worktree` | [`docs/models/qwen/fake_requant_effect_v1_en.md`](models/qwen/fake_requant_effect_v1_en.md) | `d672a5d91674` | `COPY_FROM_BRANCH` | 保留独立英文效果摘要。 |
| `docs/evaluation/qwen_2card_fake_requant_and_module_discovery_zh.md` | `evaluation_worktree` | [`docs/models/qwen/fake_requant_and_discovery_summary_zh.md`](models/qwen/fake_requant_and_discovery_summary_zh.md) | `a28e812e114b` | `COPY_FROM_BRANCH` | 保留中文汇报型概览。 |
| `docs/evaluation/qwen_2card_module_injection_surface_v1_zh.md` | `evaluation_worktree` | [`docs/models/qwen/module_injection_surface_v1_zh.md`](models/qwen/module_injection_surface_v1_zh.md) | `dd1e8bad5280` | `COPY_FROM_BRANCH` | 保留中文模块注入面说明。 |
| `profiles/discovery/qwen_2card_injection_surface_v1.md` | `evaluation_worktree` | [`docs/models/qwen/module_injection_surface_v1_zh.md`](models/qwen/module_injection_surface_v1_zh.md) | `0ae5f6594834` | `COPY_FROM_BRANCH` | 机器生成英文 profile 仍在分支；canonical docs 提供中文人工阅读入口。 |
| `profiles/discovery/qwen_2card_injection_surface_v1.json` | `evaluation_worktree` | [`profiles/discovery/qwen_2card_injection_surface_v1.json`](../profiles/discovery/qwen_2card_injection_surface_v1.json) | `—` | `COPY_FROM_BRANCH` | 机器可读 profile 原样复制并校验 SHA-256，供 evaluation/discovery 直接读取。 |
| `fault20_fast_validation/reports/fault20_final_architecture.md` | `canonical` | [`docs/faults/fault20/architecture.md`](faults/fault20/architecture.md) | `03440d1bb145` | `KEEP_CANONICAL` | 原报告保留为 source/evidence pointer；新页给出 canonical 阅读入口。 |
| `fault20_fast_validation/reports/fault20_capability_matrix_final.md` | `canonical` | [`docs/faults/fault20/capability_status.md`](faults/fault20/capability_status.md) | `92d67c082f3c` | `KEEP_CANONICAL` | 原 capability matrix 不移动，避免破坏历史路径。 |
| `fault20_fast_validation/reports/fault26_fia_writer_lossless_phase_c_20260818.md` | `canonical` | [`docs/faults/fault26/validation_status.md`](faults/fault26/validation_status.md) | `2eab4b3c304a` | `KEEP_CANONICAL` | 原验证报告保留；新页只说明证据边界。 |
| `fault20_fast_validation/reports/fault26_qmm_residual_physical_adjacency_audit_20260818.md` | `canonical` | [`docs/faults/fault26/architecture.md`](faults/fault26/architecture.md) | `53f39f03f9da` | `KEEP_CANONICAL` | 原物理邻接审计保留；新页强调生命周期与物理地址区分。 |
| `fault20_fast_validation/reports/fault39_matmul_fault_reproduction_summary.md` | `canonical` | [`docs/faults/fault39/validation_status.md`](faults/fault39/validation_status.md) | `be64ad2f7203` | `KEEP_CANONICAL` | 原复现总结保留；新页不升级 accumulator-root-cause claim。 |
| `fault15_scale_granularity/fault15_scale_granularity_trace_report.md` | `canonical` | [`docs/faults/fault15/validation_status.md`](faults/fault15/validation_status.md) | `bf1a4d30d18c` | `KEEP_CANONICAL` | 仅建立入口，不新增 Fault15 capability。 |
| `experiments/fake_requant_reference/README.md` | `fault_research_worktree` | [`docs/models/qwen/fake_requant_effect_v1.md`](models/qwen/fake_requant_effect_v1.md) | `15feecae775e` | `COPY_FROM_BRANCH` | 参考研究分支的冻结配置/实验边界；效果数值仍以 evaluation evidence 为准。 |
| `fault20_fast_validation/reports/fault20_executor_cache_preload_discovery_20260805.md` | `canonical` | [`docs/archive/fault20/README.md`](archive/fault20/README.md) | `b32116a76514` | `ARCHIVE` | 不物理移动 evidence-linked 历史探查；archive catalog 指向原路径。 |
| `fault20_fast_validation/reports/fault26_residual_writer_timeline_audit_20260818.md` | `canonical` | [`docs/archive/fault26/README.md`](archive/fault26/README.md) | `1a791091b338` | `ARCHIVE` | 不物理移动过程型 writer timeline 审计。 |
| `fault20_fast_validation/reports/fault39_a3_validation_handoff.md` | `canonical` | [`docs/archive/fault39/README.md`](archive/fault39/README.md) | `db5f80aa2ab0` | `ARCHIVE` | 不物理移动 handoff 报告。 |
| `fault20_fast_validation/evidence/README.md` | `canonical` | [`docs/README.md`](README.md) | `992c48bb51e2` | `KEEP_CANONICAL` | docs 只建立 evidence pointer，不迁入 evidence。 |
| `（本轮新增）` | `canonical` | [`docs/evaluation/fault_effect_evaluation_core_metrics_zh.md`](evaluation/fault_effect_evaluation_core_metrics_zh.md) | `—` | `KEEP_CANONICAL` | 本轮新增的一级评估逻辑文档。 |
| `（本轮新增）` | `canonical` | [`docs/faults/fault41/validation_status.md`](faults/fault41/validation_status.md) | `—` | `KEEP_CANONICAL` | 本轮只固化已知 MSProbe 观察限制，不导入 runtime/evidence。 |

## 规则

- `KEEP_CANONICAL`：原路径保持为 source/evidence；新文档只提供 canonical interpretation 或入口。
- `MOVE`：旧 docs 路径留下 redirect notice，新路径为正式维护入口。
- `COPY_FROM_BRANCH`：保留来源分支的独立价值，并在 docs branch 创建稳定阅读入口。
- `ARCHIVE`：过程型资料不物理移动，archive catalog 定位原报告，避免破坏 evidence/hash/引用。

完整 Markdown 盘点和逐字重复检测见 [document_inventory_v1.md](document_inventory_v1.md)。
