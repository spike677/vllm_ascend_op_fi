# Fault26 workspace/buffer stomp：只读路径探查计划

## 当前判定

Fault26 适合复用 Fault20/39 的 executor-cache metadata 与动态 Run 地址证明框架，但第一版只做只读路径探查：

```text
FIA workspace
  -> candidate victim storage
  -> ReduceSum input/output or reused storage
  -> residual consumer
```

本阶段明确不做：

- 缩小 FIA workspace；
- 触发真实 out-of-bounds write；
- 写入任何 activation、residual、KV cache 或 workspace；
- 用 tensor 名称或历史 `add_3` 名称替代动态 storage 证明；
- 声称 `FIA_REAL_OUT_OF_BOUNDS_WRITE_PROVEN`。

## 1. 已完成的本地审计

当前公开源码中的 typed adapter 只覆盖：

```text
aclnnQuantMatmulWeightNz
aclnnMatmulWeightNz
aclnnMatmul
aclnnAscendQuantV3
```

尚无 FIA、ReduceSum 或 residual consumer 的 typed adapter。现有 Fault39 runtime 已经提供可复用的四个底层点：

```text
AddParamToBuf(const aclTensor*)
AddParamToBuf(aclTensor*)
AddOpExecCache(cache)
OpExecCache::Run(workspace, stream, addresses)
```

为了不改动已经验证的 Fault39 数值逻辑，本轮新增独立目录：

```text
fault20_fast_validation/source/fault26_readonly_trace_v1/
```

## 2. 官方接口边界

CANN 文档确认：

- `aclnnFusedInferAttentionScoreV3` 是标准两段式接口，execute 阶段接收 workspace pointer、workspace size、executor 和 stream；
- V3 文档列出的支持范围包含 A2，而当前文档页面未把 A3 列为支持目标；
- `aclnnReduceSumGetWorkspaceSize` 的输入为 self、dims、keepDims、dtype、out，并返回 workspace size/executor；execute 同样接收 workspace pointer、size、executor 和 stream。

因此 A2 第一轮可以读取 workspace pointer/size，但不能把 API、shape、slot 或支持范围外推到 A3。
公开文档只用于确定两段式接口边界；最终 ABI 仍以 A2 CANN 8.5.1 镜像内头文件和真实导出符号为准。

## 3. 已实现的只读 tracer

新增组件：

| 文件 | 作用 |
|---|---|
| `fault26_trace_preload.cpp` | interpose executor-cache 四个边界，采集 metadata、动态 Run 地址、API、stream 与时序 |
| `fault26_opapi_trace_proxy.cpp` | 只转发 FIA V3/ReduceSum execute，并把 workspace size 传给 tracer |
| `fault26_trace_shared.h` | execute kind 与 bridge ABI |
| `analyze_fault26_trace.py` | 离线统计 API、slot proof、workspace overlap 与跨算子地址关系 |
| `build.sh` | 版本锁定、符号检查、ABI smoke 与只读性检查 |
| `README.md` | 两阶段部署与停止条件 |

### 3.1 门禁

只有以下两个变量同时为 `1` 才采集：

```text
VLLM_SERVER_DEV_MODE=1
FAULT26_TRACE_ENABLED=1
```

runtime 还要求真实 `OpExecCache::Run` 来自显式配置的 CANN 路径。若 ABI、路径或 metadata getter 不完整，则不采集并原样调用真实 Run。

### 3.2 只读性

runtime 不引用：

```text
aclrtMalloc*
aclrtMemcpy*
aclrtMemset*
```

构建脚本会对动态符号再次检查并拒绝异常产物。tracer 不 D2H，不读取 tensor 内容，不修改 addresses vector，不同步 stream，不访问 executor 私有布局。

### 3.3 两阶段运行

第一阶段只加载 runtime：

```text
TRACE_ALL_APIS=1
TRACE_CAPTURE_TENSORS=0
```

目标是确认当前模型实际调用的是哪一种 FIA/ReduceSum/Add 或融合 consumer。只有确认真实库导出 V3 和 ReduceSum 后，第二个独立进程才加载 scoped opapi proxy，并开启 tensor metadata：

```text
TRACE_ALL_APIS=0
TRACE_CAPTURE_TENSORS=1
```

这样不会因为 proxy 自己导出一个不存在的符号而改变框架的 capability detection。

## 4. 证据字段

每个目标 Run 记录：

```text
pid / tid / device
monotonic sequence / timestamp
GetLogApiInfo result
execute API
cache / run index
workspace pointer / known size
stream
formal tensor ordinal
dtype / format / shape / stride / offset / bytes
compact Run slot candidate
slot_proven
dynamic Run address
intra-Run address alias
workspace overlap / gap
real Run status / elapsed enqueue time
```

不记录完整 tensor 内容。

## 5. 当前仅完成的验证

Windows 本地没有 Ascend/CANN aarch64 headers 与编译器，因此当前只完成：

```text
Python analyzer syntax                  PASS
synthetic FIA -> ReduceSum relation     PASS
synthetic ReduceSum -> Add relation     PASS
source-level no-write API audit         PASS
CANN aarch64 compile                    NOT RUN
A2 ABI/load smoke                       NOT RUN
A2 model API inventory                  NOT RUN
A2 victim mapping                       NOT PROVEN
```

合成 JSONL 只验证分析器，不属于模型或算子 E2E 证据。

## 6. A2 严格验证顺序

在获得明确 SSH 私钥路径或用户手动进入服务器后，严格串行：

1. 只读确认 A2 容器、服务、CANN 路径和空闲设备，不启停现有服务；
2. 将新增源码复制到独立目录；
3. 在与目标服务相同的镜像中编译，检查 real symbols、SONAME、NEEDED、RUNPATH 和无 device-write symbols；
4. 新建独立 tracer-only 服务进程，不与 Fault20/39 数值 runtime 共载；
5. 固定请求执行无 tracer baseline；
6. runtime-only 全 API inventory；
7. disarm/退出实验进程并确认 baseline 精确恢复；
8. 只有确认 V3/ReduceSum 后才启动 targeted proxy 进程；
9. 证明 FIA workspace、candidate victim、ReduceSum 与 residual consumer 的时序和地址关系；
10. 保存结论 Markdown，JSONL 只留实验机。

## 7. 本阶段 PASS 条件

必须同时满足：

```text
exact API observed
workspace pointer and requested size observed
target cache registration present
formal metadata complete
dynamic compact slot proven
producer/storage/consumer time order proven
exact address reuse or interval relation explicitly classified
baseline response/token unchanged
worker/health normal
```

如果实际模型没有独立 ReduceSum 或 Add，记录 fusion 事实并回到 typed consumer discovery，不能用历史 `add_3` 直接替代。

## 8. 后续能力门禁

只有只读 mapping PASS 后，才允许设计独立的 lossless write-back 实验。第一档仍应是：

```text
read victim -> write identical bytes to same proven legal range -> E2E exact equivalence
```

在 lossless 门禁之前，不实现任何 overwrite/random/bitflip/NaN/Inf/stale-copy 模式。即使后续安全 proxy 成功，准确结论也只能是：

```text
FIA_TRIGGERED_VICTIM_BUFFER_CORRUPTION_PROVEN
RESIDUAL_CORRUPTION_PROPAGATION_PROVEN
MODEL_OUTPUT_FAILURE_REPRODUCED
```

不能写成：

```text
FIA_REAL_OUT_OF_BOUNDS_WRITE_PROVEN
```

## 参考

- [CANN 8.5 FusedInferAttentionScoreV3](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/API/aolapi/context/aclnnFusedInferAttentionScoreV3.md)
- [CANN ReduceSum 接口原型](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/API/aolapi/context/aclnnReduceSum.md)

## 9. 2026-08-14 A2 实机只读审计结果

### 9.1 SSH 与服务状态

- 已恢复并验证 Windows 私钥：`C:\Users\admin\.ssh\npu_codex_ed25519`；公钥指纹为 `SHA256:4sOLIA+tRTlOBEOGS+K8OJceXk5omriw3fJM67LErnc`。
- SSH 配置别名 `npu-dual` 可在 `BatchMode=yes` 下免密连接。
- 目标模型为 `/AIdata/haoran/Qwen3-32B-w8a8-nopdmix`，服务名 `qwen3-32b-w8a8`，端口 `18005`，TP=2，使用 NPU 2/5。
- 主服务 PID `148458`，TP worker PID `152654/152655`；最终回归时三者均保持存活，NPU 2/5 仍分别由原 worker 占用。
- `/health` 返回 HTTP 200。
- 现有 Fault39 控制面为 `enabled=0`，本轮没有 arm、触发或修改任何既有故障模式。

服务位于遗留容器 mount namespace 中，容器 ID 已不被当前 Docker daemon 识别；`/fault39` 映射到宿主机：

```text
/home/haoran/fault39_dev_20260812
```

因此本轮明确保留现有推理进程，不执行 stop/restart，也不把新库加载到现有 worker。

### 9.2 A2 CANN 8.5.1 ABI 证据

目标环境为 aarch64，`torch_npu=2.9.0.post1+gitee7ba04`。镜像内 CANN 8.5.1 的真实头文件和 `libopapi_real.so` 已确认导出：

```text
aclnnFusedInferAttentionScoreV3
aclnnFusedInferAttentionScoreV3GetWorkspaceSize
aclnnFusedInferAttentionScoreV3GetMaxWorkspaceSize
aclnnReduceSum
aclnnReduceSumGetWorkspaceSize
```

其中 V3 FIA 和 ReduceSum execute 均为标准的 workspace、workspace-size、executor、stream 四参数执行接口。`GetMaxWorkspaceSize` 虽由真实库导出，但未在本轮公共头文件中取得可安全依赖的正式原型，因此 tracer 不 interpose、不猜测该私有/扩展 ABI。

静态源码审计确认当前 vLLM-Ascend attention 路径包含 fused infer attention 调用，但这不等价于已证明本次请求进入外部 `aclnnFusedInferAttentionScoreV3`。同时 residual 路径可能被融合为 `npu_add_rms_norm`，不能把历史图名 `add_3` 当作当前模型的独立 consumer。

### 9.3 编译、符号与独立装载验证

源码复制到独立目录：

```text
/fault39/source/fault26_readonly_trace_v1
```

产物写入独立目录：

```text
/fault39/runtime/fault26_readonly_trace_v1
```

在现有容器 mount/PID/network namespace 内、但不进入现有 vLLM 进程完成 aarch64 编译。以下符号检查 PASS：

```text
op::internal::OpExecCache::Run(...)
op::internal::AddOpExecCache(...)
op::internal::AddParamToBuf(const aclTensor*)
op::internal::AddParamToBuf(aclTensor*)
aclnnFusedInferAttentionScoreV3
aclnnReduceSum
```

构建门禁再次确认 runtime 不引用 `aclrtMalloc*`、`aclrtMemcpy*` 或 `aclrtMemset*`，也不调用 `aclOpExecutor::GetOpExecCache()`。

产物 SHA-256：

```text
0cf65d261c51731cfd2f19820daf342d31fa28c81888ff5aba47de5336d33062  libfault26_readonly_trace_abi0.so
b00e89f51247482cf46590f20f799d8751f55b94cf198d654b5afaa35c4716ae  libopapi.so
234b7582c6c8866001dc8be727c2572e4fe5a4636ec1edd7c1ca143c83ee0996  real/libopapi_real.so
```

独立 Python 进程加载 `libnnopbase.so` 和 tracer 后，显式 self-test 产生 resolution、execute-enter、execute-exit 三条 JSONL。关键字段为：

```text
abi_ok=true
developer_mode_ok=true
trace_enabled=true
real_library=/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libnnopbase.so
expected_fragment=/cann-8.5.1/
```

这证明 A2 上的 ABI 解析、CANN 路径锁、开发者模式门禁、日志写入与离线分析器可工作；self-test 没有执行真实模型算子，因此分析器正确报告 `runs: 0`，不能把它当作 FIA/ReduceSum 模型调用证据。

### 9.4 模型无回归证据

对同一个固定请求执行编译前双次 baseline 与编译后回归：

```text
temperature=0
seed=1234
max_tokens=32
```

三次 response 内容一致，usage 均为：

```text
prompt_tokens=17
completion_tokens=32
total_tokens=49
finish_reason=length
```

最终请求 HTTP 200，worker PID 和 NPU 占用未变化。新 tracer 没有加载到模型进程，因此这一结果证明本轮审计/编译没有扰动服务，不是 tracer-on-model 的 token 等价证据。

### 9.5 准确完成度结论

```text
SSH key recovery / passwordless access       PASS
A2 service/CANN/container read-only audit    PASS
fixed baseline determinism                   PASS
CANN 8.5.1 aarch64 compile                   PASS
symbol/SONAME/NEEDED/RUNPATH checks           PASS
no device-write API gate                     PASS
independent ABI/developer-mode/log self-test PASS
existing service no-regression               PASS
A2 model API dynamic inventory               NOT RUN
FIA workspace runtime observation            NOT PROVEN
ReduceSum runtime observation                 NOT PROVEN
FIA -> victim -> consumer address mapping     NOT PROVEN
any write/corruption/OOB experiment           NOT IMPLEMENTED / NOT RUN
```

当前可以准确命名为：

```text
FAULT26_READONLY_TRACER_BUILD_PASS
FAULT26_A2_ABI_GATE_PASS
FAULT26_INDEPENDENT_TRACE_SELFTEST_PASS
```

不能命名为：

```text
FAULT26_MODEL_PATH_MAPPING_PASS
FIA_TRIGGERED_VICTIM_BUFFER_CORRUPTION_PROVEN
FIA_REAL_OUT_OF_BOUNDS_WRITE_PROVEN
```

动态模型 inventory 需要在一个可安全启动和回收的新实验容器中，通过进程启动时加载 `LD_PRELOAD` 完成。由于当前服务所在遗留容器无法由现有 Docker daemon 安全复建，本轮不以停止健康服务为代价推进该步骤。
