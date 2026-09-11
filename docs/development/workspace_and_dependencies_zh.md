# 工作区、依赖与执行环境

## 唯一源码入口与Git关系

canonical为 `D:\project\demoncatpfi`，VS Code只打开[相对单根workspace](../../demoncatpfi.code-workspace)。当前使用 `codex/source-baseline-v1`，main和开发分支保留历史，不强行合流。旧C盘clone仅LEGACY_READONLY_SOURCE，不能复制其.git拼接仓库。

其他linked worktree继续由Git管理；当前被测Qwen修改已冻结成f67b418。旧main的两个未提交Fault26文件与研究工作树的独有内容不合入本版；保存记录见[发布盘点](../releases/source_baseline_v1/README.md)。不要手删worktree或用reset --hard清理。

## 依赖清单

| 依赖 | 归属／版本和用途 |
| --- | --- |
| src/ascend_fi、tools、profiles | 本仓库固定发布提交 |
| fake_requant_reference_extension.py | 本仓库experiments参考模块；来源1d715725，SHA560978…；实际完整值见[依赖页](../../experiments/fake_requant_reference/README.md) |
| Python | 推荐现有Python≥3.10供客户端/离线工具；实测容器3.11。单独starter避免了旧host缺shlex.join，不代表所有工具支持3.7 |
| torch / torch_npu | Qwen0.6B实际2.9.0+cpu / 2.9.0；Linux/NPU环境提供，不在源码包嵌入 |
| vLLM / vLLM-Ascend | 实测0.18.0+empty / 0.17.0rc2.dev143+gf83cb0e6d；构建标识不按镜像tag推断 |
| CANN／driver | 实测CANN8.5.1；910B4单物理卡5、driver25.5.2；换构建重新验证 |
| transformers/tokenizer | 用现有模型目录local_files_only；本次不网络下载 |
| Docker、本地镜像、npu-smi、设备节点 | Linux宿主机已有环境；starter不拉镜像、不停止其他容器 |
| 固定MT-Bench source suite | [公共配置](../../profiles/evaluation/qwen06b_demo_source_suite.json)，含原题与revision/hash，不需要旧E1/R1结果 |

源码hash不等于环境lockfile。没有声明任意版本组合都兼容，也没有承诺一条pip install部署完成。

## 将源码部署到Linux

在Linux把**同一发布源码**放到一个明确目录，例如 `/home/gyl/demoncat_source_baseline_v1`；不要把Windows绝对路径写进Linux配置。至少需要src、tools、profiles及固定reference。建议保留完整源码及发布manifest用于审计。

```bash
# Linux宿主机；已有本地代码，不在本轮执行
export DEMO_CODE=/home/gyl/demoncat_source_baseline_v1
export DEMO_ROOT=/home/gyl/demoncat_demo
export DEMO_SESSION="$DEMO_ROOT/run_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$DEMO_SESSION/reference"
cp "$DEMO_CODE/experiments/fake_requant_reference/source/fake_requant_reference_extension.py" "$DEMO_SESSION/reference/"
sha256sum "$DEMO_SESSION/reference/fake_requant_reference_extension.py"
```

核对固定SHA后按[单卡手册](../inspect/qwen06b_single_npu_user_walkthrough_zh.md)启动。starter给worker的PYTHONPATH是 `source_root/src:session/reference`，并设置参考SHA环境变量。加载文件缺失/hash不符即拒绝，不回退其他研究worktree。

模型、代码只读挂载，session结果可写；密钥只保存在session/access.token，不入Git。客户端--deployment必须显式给出；不传仍会走历史owner，这是保留的旧兼容入口，不是新用户默认方式。

## 安全和失败边界

先确认资源和项目NPU_LOCK归属；本地单服务并不自动排除其他外部客户端。每次执行先核实clean/idle/私有鉴权，同epoch观察和IDENTITY。实际全请求window没有独立scheduler observer，本轮不得升级结论。

guard拒绝后不删证据marker继续。hook移除失败、启动早期留锁、stop重复调用等已知限制列于[算法](../architecture/algorithms_zh.md)。本轮只收束源码，不修生命周期；相关改动必须单独issue/提交与实机回归。

## 离线开发

运行[测试](../../tests/README.md)与新source audit helper。不要运行会更新旧manifest的historical audit工具，不下载新依赖来伪造缺失Linux检查。源码hash核对以发布manifest为准；历史raw路径hash保持，另存映射。没有NPU也能阅读、检查CLI和评分器，但不能由测试替身认证新模型。
