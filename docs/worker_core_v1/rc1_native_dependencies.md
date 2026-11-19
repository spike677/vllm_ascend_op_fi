# RC1 Native 依赖交接

本说明属于 `demoncatpfi-yellow-rc1-20261119` 的源码交付。首批普通任务是
`39.output_proxy`、`quant.batch_shared_scale`、`18.api_half`。下述构建入口已经存在，
不新增 Native 实现，不启动模型或验收请求。**Git 源码包不含 Native 二进制、CANN、镜像或模型权重。**
黄区应在已有、获准使用的依赖环境中按配方构建，或通过其内部制品渠道取得附完整 manifest 的自有构建包。
不能只复制一个同名 `.so`，也不能用蓝区的路径或 SHA 代替黄区实际文件核验。

## 三个 Python 发行单元

安装同一 RC 源快照的全部三个发行单元；版本保持不变：

| 构建根（相对于导出包 `sources/`） | 发行单元 | 版本 |
|---|---|---|
| `.` | `demoncatpfi` | `0.2.0` |
| `src/` | `demoncat-worker-extension` | `0.1.0` |
| `fault39/` | `demoncat-fault39-core` | `0.1.0` |

Python 要求为 3.10 及以上。各包未自动安装整套 PyTorch/CANN/vLLM 运行环境；使用已审核的服务环境，
不要为通过安装而任意升级这些组件。构建 Native 时还须保留完整导出源码树，不能只给 manager 一个
`site-packages` 路径。尤其 Batch 的独立构建脚本及 CPU 测试源在源码包内，不属于运行时导入的实验 runner。

黄区已有 `demoncat` 命名空间，安装前后应核对实际 `demoncat.runtime`、Worker extension、
`fault_injection`、`demoncat_fault39` 的模块来源；不得覆盖或混入旧主项目同名模块后声称完成接线。
普通 provider 不导入本文的构建脚本。实际主项目接线仍待黄区完成。

## 已验收部署的 Native 身份

以下为蓝区实际使用的参考身份，**不是新构建输出必须伪装成的身份**。重建应产生自己的收据、SHA 和构建键，
与本次安装源码和实际 SDK 重新核对。相同版本号不保证相同 ABI/来源；CPU 构建通过不授予 NPU 验收。

| 项目 | 39.output_proxy | Batch 共享尺度 |
|---|---|---|
| 实际普通请求验收 | Source116，normal/F/OFF/health | Source118，normal/reference/lossless/calibration_off/F/OFF |
| Native 来源 | Source101 的 public ACL postwrite | Native 编译提交 `46cd8ad153ad07ce1057cfcbac73d7d9fb76e226` |
| 库 | `libdcat_postwrite.so` | `runtime/vendor/op_api/lib/libcust_opapi.so`，SONAME `libdcat_dq_group_v1.so` |
| 库 SHA256 | `9261d0867f822c90d0d179457e41bc9a2b132bc39e643b61fe5d47a3fd81424f` | `6eb6c19cff6006bc9cd0b1072504dc5d6b7db3a0d7b393af7c88f4db1c8bd2fd` |
| 构建键 / manifest | build key `05e6a510efa3a8ac8f4a7c3f77056af2027137a3d90440cc36ec31f90af1ccbf` | `build.json` SHA256 `b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac` |

参考环境是 Linux aarch64、CANN 8.5.1、Ascend910B4/A2，实际镜像 ID
`sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`。
发行标签 `quay.io/ascend/vllm-ascend:v0.18.0` 仅供识别，不能代替实际镜像 ID 检查。
RC1 不把这些结果扩展为所有 CANN 8.x/9.x、A3、任意模型或任意 Torch/vLLM 版本的通过结论。
事实来源分别见 [39 独立总结](output39_source116_single_task_summary.json)、
[Batch 独立总结](review_quant_ordinary_source118.json) 和 [18 独立总结](review_rms18_ordinary_source126.json)。

## 39：现有 public ACL 构建配方

使用 `fault_injection.native_build.NativeBuildManager`，配方
`postwrite.output.acl.v1`，脚本 `fault_injection/native_build/build_postwrite.sh`。
这不是 `39.matmul_range` 的 OpCache 库。配方里的 `aten.mm.default` 是构建配方选择键，
不表示实际模型只能使用 aten；Source116 的真实 producer 是 `vllm.unquantized_gemm.default`，
其支持仍由普通 provider 的实际源/对象/consumer 检查决定。

由部署负责人设置以下路径：`SOURCE` 为解包的完整 `sources/`，`CACHE` 为新构建缓存，
`ASCEND_HOME_PATH` 指向当前准确 SDK。已有 SDK 的环境脚本必须可用。依赖包括 aarch64 GCC/G++、
bash、nm、readelf、patchelf、Python、`acl/acl.h` 和真实 `libascendcl.so`。
manager 会记录 SDK 头、工具链组件、库、源和环境脚本身份；工具缺失或 SDK 选择含糊时拒绝。

```sh
python3 -m fault_injection.native_build --repository "$SOURCE" --cache "$CACHE" \
  inspect --mechanism 39.output_proxy --operator aten.mm.default
python3 -m fault_injection.native_build --repository "$SOURCE" --cache "$CACHE" \
  build --mechanism 39.output_proxy --operator aten.mm.default --timeout 1800
```

从真实 build 返回值取得 `artifact_directory`，设为 `ARTIFACT`，保留 `inputs.json`、
`receipt.json`、`build.log` 与整个 `native/`；不得手工补成功字段。随后执行：

```sh
python3 -m fault_injection.native_build --repository "$SOURCE" --cache "$CACHE" \
  verify --artifact "$ARTIFACT"
```

部署层经 `NativePostwriteLibrary(actual_library, actual_sha256)` 加载本次已验证库，
再安装 `runtime_only=True` 的 `install_fault39_output_serving_binding`。
库的磁盘字节、实际映射 inode、ABI、read-span 和租约容量仍由现有代码检查；
普通路径最多一个 live lease，原 finite-before 检查、实际 consumer、restore/release 不能删除。
不要把新缓存的完整源核对失败“修复”为旧 Source101 manifest；重建与重核是正确分支。

## Batch：独立 group Native 构建配方

使用现有 `experiments/c2c3_closure_20261117/providers_quant_build_group.py`。
它在固定镜像内执行无设备 SDK 编译和 CPU 合同；有管理权限的部署负责人执行，不由运行中 provider 创建容器。
要求 aarch64 宿主、Docker、上述已存在的精确镜像及其中 CANN 8.5.1 SDK。
脚本不下载镜像，不挂 NPU，关闭网络，仅清理它自己的 build 容器。
`NATIVE_OUT` 必须是不存在的新目录，`SOURCE_COMMIT` 为本 RC 的完整实际提交标识：

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_build_group.py" \
  --source "$SOURCE" --source-commit "$SOURCE_COMMIT" --out "$NATIVE_OUT" \
  --image-id sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0
```

生产源为 `batch_scale_group_core.h`、`batch_scale_group_runtime.h`、
`batch_scale_group_runtime.cpp`，均位于 `fault_injection/native_adapters/`。
脚本还编译实际 SDK ABI 与有明确标注的 CPU loader doubles，记录动态库、导出、SDK ABI、
stock `libopapi.so`、CPU 合同、镜像和完整构建收据。保留整个输出目录；
实际 `build.json` SHA 是后续 `install_quant_projection_binding(..., build_sha256=...)` 的绑定值。
这个构建入口不使用 `NativeBuildManager` 的其他旧量化库，也不能用 GMM2/metadata-only observer 替代。

普通 Worker 启动前须保留现有 group vendor/stock custom 库顺序。以下是参考镜像的真实布局，
由部署层将本次 bundle 映射为 `/quant`；它不是用户层名或故障参数：

```sh
source /usr/local/Ascend/cann-8.5.1/set_env.sh
export DCAT_DQ_GROUP_V1_ALLOWED=1 DCAT_DQ_V2_ENABLE=0 VLLM_SERVER_DEV_MODE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 TASK_QUEUE_ENABLE=0
export DCAT_REAL_OPAPI=/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so
export ASCEND_CUSTOM_OPP_PATH=/quant/runtime/vendor:/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend
export LD_PRELOAD=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so
```

这是已审查的精确启动配置；不要在已加载的 Worker 中临时更换 preload，不要混入旧 C1/group 之外的库。
`QuantGroupNative.load` 会逐文件核 bundle SHA、三份生产源与已安装绑定的字节、真实 stock 库路径/SHA、
实际映射/符号 owner 和 SDK ctypes 布局；缺依赖或不一致必须保持拒绝。
库可加载不等于当前投影可注入：真实 INT8 权重、Qwen 普通动态 W8A8、TP1/B8 首次纯 decode、
输入行关系、Q/scale 双效果、真实 QMM consumer、六批当前作用域校准及 OFF 均保留。

## 18 与停止条件

`18.api_half` 不需要本仓库额外故障 Native DSO，复用当前服务实际 `npu_rms_norm` 返回和真实 consumer，
仍依赖匹配的 Torch、torch_npu、vLLM-Ascend 和 CANN 环境。Source126 只验收 Qwen3-0.6B BF16、
TP1 串行 eager fresh prefill 的一个实际 RMS 返回，normal/F/OFF/health 四请求；没有新 Native 构建需求。

SDK、镜像、库或 callable 身份不匹配时应报告未配置/不支持并停止，不能用同名文件、CPU double、
历史通过 JSON 或参数复位代替。新构建的 CPU 成功仅说明构建与相应 CPU 合同；
黄区接入后仍须在同一实际 Worker 执行该机制已有普通 normal/F/OFF/health 或 Batch 六批协议。
下一任务必须使用 fresh scope，不清零 attempted、不复用 seal 或过期地址。
本交接不授予黄区主项目接线、C1/C2/C3/E5 或新增模型/版本的验收资格。
