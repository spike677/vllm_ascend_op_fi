# 实际命令目录

默认显式传 `--deployment`，不走旧双卡E1默认owner。读[单卡手册](../docs/inspect/qwen06b_single_npu_user_walkthrough_zh.md)后按顺序执行，不能遍历运行脚本。

| 入口 | 位置／前提 | 真实参数 | 输出、副作用、下游 |
| --- | --- | --- | --- |
| [private_demo_service.py](private_demo_service.py) | Linux宿主机，Docker/本地镜像/模型/空闲NPU/参考文件 | `start/stop/status --session`；start另有 `--source-root --model --device --port --image --lock-root` | lease/鉴权/容器/deployment；stop仅本session容器 |
| [prepare_private_demo_suite.py](prepare_private_demo_suite.py) | 容器或Linux CPU，本地transformers/tokenizer | `--source-suite --model --max-tokens --out` | 选82/86/149并重算模板长度；不推理/下载，新文件 |
| [interactive_inspect.py](interactive_inspect.py) | 私有服务客户端 | `status/list/describe/observe/poll/stop/candidates/export`、`--base-url --api-key-file --args --output` | 只读查询或有限元数据hook；export不arm。用新输出文件，旧子命令不都保证拒绝覆盖 |
| [inspect_model_targets.py](inspect_model_targets.py) | 服务ready，客户端 | `--base-url --api-key-file --out`，可选 `--roles` | 结构报告，不等于绑定计划 |
| [run_model_target_plan_live.py](run_model_target_plan_live.py) | 当前容器、真实lease | `--deployment --session --suite --roles --multiplier --max-prompt-tokens --identity-probe --probe --discover-only --out` | 正常观察和关闭计划；无有损注入 |
| 同一runner | 当前epoch绑定计划 | `--deployment --session --suite --plan --identity-probe --probe --out` | IDENTITY、每题B/F/R、finally清理、raw/result；独立新目录 |
| [summarize_private_demo.py](summarize_private_demo.py) | 离线Python、完整evaluation目录 | `--evaluation --out` | 新中文原文报告/数值/Dashboard，不重写输入 |
| [audit_private_demo.py](audit_private_demo.py) | Linux部署环境，只读 | `--session --branch --head --out` | 源码/安装记录/证据哈希，不推理 |
| [audit_source_baseline.py](audit_source_baseline.py) | 本地Python/Git | `inventory/links/verify --out`；verify加 `--manifest` | 独立新审计文件；哈希、链接、help；不启动服务 |

## 辅助与历史入口

`private_service_owner.py`是部署适配模块而非CLI，复用 `run_normal_diagnostic_live.py` 的已有owner。`run_interactive_inspect_live.py`、`inspect_dataset_live.py`、`run_inspect_exploration_live.py`和supervisor面向旧会话，不作新服务器默认入口。

`run_distortion_evaluation.py`、`replay_fault_effect_profile.py`可离线读旧证据，不证明新配置PPL。`verify_fake_requant_release.py`检查旧冻结manifest，本次用新audit helper。`audit_documentation_v2.py`会改写旧审计输出，本轮不运行。`acquire_inspect_mtbench.py`用于网络获取，本次公共source suite已提供，不必下载。

路径建议显式Linux绝对路径，并核对host/container映射；API key只用文件传递，不记录值。`--help`通过不是NPU运行验证。失败时保留raw并按[生命周期限制](../docs/architecture/algorithms_zh.md)停下检查。
