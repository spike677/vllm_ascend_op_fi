# Native 对象区间：A3 构建准备

本文件交负责人保留，尚不是 A3 验收通过记录，也不要求黄区现在执行。冻结 r3 不包含本轮 Native 新实现。A2 的二进制 SHA 不可直接登记为 A3 产物；黄区产物须先回传构建身份并审查。

适用目标是 87 的 Ascend910_9382、CANN 9.1、0.26 `-a3` 专用实例。仅可信部署侧可以提供活跃 Tensor 对象和已审查的 DSO 路径/SHA；用户配置没有地址、文件或任意索引入口。新增导出与旧 ABI 并存，旧冻结物不改。

## 无设备构建

在准备好的隔离 `-a3` 容器内执行。仓库源码只读挂载至 `/source`，独立可写产物目录挂载至 `/build`。首次不映射 NPU，不加载模型，不修改宿主驱动/CANN。先保存镜像的真实 RepoDigest、容器内 CANN 版本文件和宿主只读 `npu-smi info` 原始芯片名；不能用镜像标签推断芯片身份。

```bash
source /usr/local/Ascend/cann/set_env.sh
python3 -B /source/tools/build_native_regions.py \
  --source /source/fault_injection/native_adapters/postwrite \
  --sdk /usr/local/Ascend/cann \
  --output /build/native-regions-a3-attempt1
```

输出目录必须不存在；失败产物和 `receipt.json` 保留，下次用新目录。工具以 `-Werror` 编译三个 CPU ACL 替身测试，之后链接真实 `libascendcl`；测试成功只证明 CPU 合约及 SDK 编译。工具不申请部署许可，不调用 NPU，不自动安装共享库。

`receipt.json` 包含源文件 SHA、编译器、全部命令及退出码、可读 SDK 版本文件、导出/依赖、二进制字节数与 SHA、开始/结束时间。若 `sdk_version_files` 为空，另行提供容器内实际版本文件的路径、原文和 SHA，不能留空登记。另附镜像 digest、实际 A3 芯片名、driver/firmware、torch/torch_npu/vLLM/vllm-ascend 版本及部署源码指纹。构建失败不进入加载或注入。

## 审查及后续运行边界

1. 核对源 SHA 与候选源码、编译/测试返回码、真实 ACL 链接依赖和 `dcat_pw_open_stream_v2`、`dcat_pw_copy_region_v2`、restore/release 导出。
2. 审查通过后，在独立实例只读挂载该库；binding 的 `native_library=[绝对路径, sha256]` 必须与部署许可中的 `native_library_sha256` 相同。许可来源仍是 `YELLOW_DEDICATED_INSTANCE`，硬件和 TP 精确匹配。
3. 当前 Native41 是 ScatterPaKvCache → 当前 decode V slot → 实际 FIA 的同 stream 窗口。直接 K/V prefill 不伪装成 KV 注入；MLA latent、共享 prefix、跨 stream 未适配则拒绝。Native26/20 不能沿用41的资格。
4. 后续验收仍需公共入口、同组 Worker 两个作用域、四阶段和每阶段四 token；逐 rank 覆盖、真实消费者设备事件、区域字节恢复、OFF/health 完整 hidden/logits 精确恢复、资源零。结果仅按实际组合登记。

本文件不赋予未验收部署执行资格，不替代稍后按 A2 实测结论编写的机制验收步骤。

## r4修正：工具与硬件身份

`tools/build_native_regions.py` 不调用、不依赖patchelf。它使用g++、nm和readelf，以 `-L<SDK>/lib64 -lascendcl` 链接；运行时在同一隔离容器加载CANN的set_env.sh，使动态链接器找到真实ACL库。无需安装patchelf，也无需修改宿主环境。若出现ACL依赖找不到，保留readelf/ldd结果和环境来源，不能复制未知库或绕过SHA。

硬件身份以已经初始化的Worker中 `torch_npu.npu.get_device_name(torch_npu.npu.current_device())` 原值为准。单卡逻辑映射的自检通常等价于 `get_device_name(0)`；多rank不能强制都读逻辑0。`Ascend910_9382`识别为A3，`Ascend910B4`识别为A2，npu-smi的泛称 `Ascend910` 不足以分类，保留未证明。deployment回执新增原设备名、逻辑设备和读取来源；族分类不等于Native构建或机制资格。

无设备构建不初始化NPU获取该名字；运行身份来自后续专用实例已有Worker，只读打印原值。构建收据、镜像digest、CANN SDK版本和Native SHA由Codex审查后，才在r4步骤中使用该产物。41、26、20各自验收，TP1/TP2分别登记。
