# CANN 8.x / 9.x：实际接口分组与验收边界

本次取消了 Fault39 的“路径不含 8.5.1 就拒绝”限制。适配按接口能力、精确构建
身份和实测结果分层，不按 `8.* / 9.*` 一次性放行。**当前构建已在实际
8.5.1、9.0.0、9.1.0、9.2.0 上完成算子级 NPU B/F/OFF 与独立统计验收；
8.5.1 另完成 Qwen3 TP1/TP2、9.1.0 完成 Qwen3 TP2 推理验收。** 历史 8.5.1
结果保留在原 evidence/reports，不能替新构建背书。

## 实际检查了哪些版本

依据 124 上安装包的 `ascend_toolkit_install.info`、头文件、导出符号和编译探针，
原件见[接口清单及构建回执](../../experiments/provider_adaptation_20261116/README.md)。

| 实际版本 | 内部版本 | Fault39 当前源码 CPU 构建 | 新构建 NPU discovery / lossless / B/F/OFF |
|---|---|---|---|
| 8.5.1 | V100R001C25SPC002B220 | 通过 | 算子通过；Qwen3 TP1/TP2通过 |
| 9.0.0 | V100R001C10SPC001B250 | 通过 | 算子通过；新版完整vLLM模型未验收 |
| 9.1.0 | V100R001C11SPC001B243 | 通过 | 算子通过；Qwen3 TP2通过 |
| 9.2.0 | V100R001C12SPC001B237 | 通过 | 算子通过；新版完整vLLM模型未验收 |
| 其余 8.x / 9.x、RC、定制包 | 本次未取得对应实际安装 | 可运行能力探测；不能声称已支持该具体安装 | 未验收 |

名为 `cann:8.5.0-910b-...` 的镜像实际安装的是 8.5.1 和 9.0.0，因此没有将它记为
8.5.0 通过。镜像标签、目录名称和版本字符串均不替代安装元数据与库指纹。

## 真正需要区别的接口

| 边界 | 本次实测差异 | 适配处理 |
|---|---|---|
| Fault39 MatMul | 四版本的 `aclnnMatmulGetWorkspaceSize` 六参数与 execute 签名相同 | 共用 `matmul-cache-abi0-v1` 源码，每个安装单独编译和绑定运行库 |
| 私有 cache/metadata | 五个必需 cache 符号及七个 metadata getter 均存在；`aclTensor=544`、`OpExecCache=320` 均相同，但头文件与库指纹不同 | 编译检查真实方法签名和 inline accessor；运行时核验实际库；不能用 sizeof 相同推断布局或 slot 语义一致 |
| `AddRmsNorm` | 8.5.1 的 `yOut/rstdOut/xOut` 为 `const aclTensor*`；9.0/9.1/9.2 为 `aclTensor*` | 需要按实际声明选择 const-output / mutable-output 包装器；这是源代码类型差异，不直接宣称指针的 C 调用约定发生变化 |
| 旧 Fault26 `AddRmsNormBias` | 四个标准 toolkit 均未找到；本轮在vLLM-Ascend自定义`libcust_opapi.so`找到实际实现 | 26使用匹配自定义库和`_C_ascend.npu_add_rms_norm_bias`，核验真实残差语义；没有别名到标准`AddRmsNorm` |
| FIA API 代际 | V1/V2/V3 的 GetWorkspaceSize 分别有 32/41/44 个参数；各同名代际在四版本间的声明一致 | 按实际符号选择对应 typed wrapper，不把不同 API 代际互相强转 |
| DynamicQuantV2 / QuantMatmulWeightNz | 分别 9/15 参数，四版本间的参数类型一致 | 可以共用声明适配，但量化语义、私有 tiling 和实际命中必须单独验证 |
| 动态库布局 | 四版本均有主 `libopapi` 以及 `libopapi_nn` / `libopapi_transformer` 分库 | 构建自动识别可用分库；Fault39 包装 main 与可用的 nn。main-only 分支只有 CPU 合同测试，没有取得旧版本 NPU 通过证据 |
| 33 的 host/kernel tiling | 公共 API 一致不能证明私有 tiling 结构、模板和 kernel 可跨版本复用 | 必须匹配工具链、源码/构建、芯片和实际分支；本轮 Fault39 通过不提升 33 的支持状态 |

9.2 的依赖头文件还提示 `exe_graph/runtime/tensor.h` 将在 2027-06 后移除；当前仍能
编译，这不是目前必须拆分 Fault39 算法实现的依据。

CANN 官方将 op_cache 接口标为可能改变的保留接口，这也是精确构建绑定与逐安装
实测的依据：[CANN 9.1 op_cache 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/aolapi/atlasascendc_api_07_1115.html)。

## 实现与部署

- `native/abi_probe.cpp`：以所选安装的真实头文件检查 MatMul、Run、AddOpExecCache、
  AddParamToBuf、枚举和 metadata accessor；输出 CPU ABI 事实，不加载 CANN。
- `tools/version_profile.py`：记录实际安装元数据、头文件和源码 hash、必需符号、
  原始库与私有 SONAME 副本的身份，以及匹配的 `set_env.sh`。
- `native/runtime_identity.h`：对已加载库的可执行段及磁盘完整文件计算 SHA256。
  缓存成功检查并保持该映射存活；要求工作进程期间库不可变。此检查是构建一致性，
  不是 ABI 正确性的证明，也不是数字签名。
- `native/runtime.cpp`、`metadata_proxy.cpp`：替换版本路径判断；runtime 与代理必须
  使用同一 profile。身份不符无法配置注入，代理不会按错误头文件读取 metadata。
- `tools/deployment_profile.py`、`run_124.py`：先在无设备的 CPU 容器中核对文件，显式
  source 匹配环境，检查 `ASCEND_HOME_PATH`、`ASCEND_OPP_PATH` 和实际库搜索顺序，
  再经过资源锁及每例门禁。现有 9.0 镜像包含 8.5.1 残留环境，不能直接沿用它。
- `vllm_validate.py`：lossless 回执绑定各 rank 的 `build_profile_sha256`，旧版本回执
  不能为另一个构建打开 pattern 验收门槛。

示例（在对应 aarch64 CPU 构建环境中，不占 NPU）：

```bash
FAULT39_ASCEND_TOOLKIT=/usr/local/Ascend/cann-9.1.0/aarch64-linux \
FAULT39_OUTPUT_DIR=/work/build-unique-attempt \
bash fault39/native/build.sh
```

输出包括 runtime、main/可选 nn proxy、`version_profile.json`、`build_manifest.json`、
`abi_facts.json` 和 CPU identity 测试结果。`run_124.py --image ...` 选择与该构建匹配的
部署镜像。模型 runner 已适配实际 vLLM 0.18/0.23，仍检查实际 scheduler、worker
签名和 rank；并未泛化到所有 vLLM 版本。0.23 在本次9.1镜像中必须显式选择
`block_size=128`，关闭 chunked prefill 和 flashcomm1；默认 block16 初始化失败
的尝试保留。这是模型部署差异，不是 Fault39 算法或 CANN Matmul API 的变化。

## 当前验收状态

四版本分别使用当前 `source-a5` 的独立 `build-*-a5`。均通过真实 Run 发现、
self-only/mat2-only/dual lossless、实际有限输入替换、OFF恢复和原输入完整storage哈希
核验。原始 `statistics/mix` 的独立首差均为
`Torch.matmul.0.forward/output/0` 的 Max：`-0.3359375 → +Inf`。
固定容差为 `rtol=1e-5, atol=1e-8`；此前的 Mean/Norm null 保留不可观测。

8.5.1 Qwen3 TP1与TP2分别绑定自身lossless回执，真实推理中的自动选点、命中与
恢复通过。实际step0未发现可观测统计差异，step1首差在
`Functional.linear.3.forward/output/0`；TP2两rank的Max分别为
`0.625 → +Inf`、`0.369140625 → +Inf`。这不是训练链路，也不是每层全部注入。
原生BF16抵消机制没有变成输出覆写；仍不声称定位了kernel内部accumulator类型。

9.1.0 使用同一 `build-91-a5` 与 Python `source-a7` 完成独立TP2 lossless、B/F/OFF。
两rank均实际命中并恢复；step0没有可观测统计差异，step1同样首先在
`Functional.linear.3.forward/output/0` 观察到Max `0.625 / 0.369140625 → +Inf`。
原生回执的 `MSPROBE_ANALYSIS_PENDING` 是采集结束时状态，后续独立报告另存，
没有改写原回执。9.0/9.2未运行完整模型，不能沿用9.1的通过结论。

此前a1在NPU进程中因加载库身份解析而拒绝；修正为核验实际已加载的
`libascendcl.so`后，以全新构建重新完成验收，未关闭身份门禁。旧失败目录保留。

其余五条provider的支持版本和模型范围独立列在[本轮总结](../../experiments/provider_adaptation_20261116/README.md)。
Fault39四版本通过不让20/26/33自动跨版本通过。本轮按用户要求只公开实现、版本事实
和总结，不上传新增的原始statistics、完整张量、模型权重或vendor库。
