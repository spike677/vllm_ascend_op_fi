# CANN、实际加载与硬件兼容矩阵

本轮区分三种来源：**124宿主目录、独立镜像内的真实安装、实际Worker已加载库**。
镜像标签不是安装版本，宿主`cann-9.0.0`目录名称也不替代安装元数据。
新只读盘点与此前四版本精确构建分别保存在
[environment_matrix.json](../experiments/overnight_20261116/environment_matrix.json)和
[profiles/environments](../profiles/environments)。四版本profile原件SHA均与已发布构建清单一致。

## 实际硬件

124本轮`npu-smi info`观测到物理卡2/5均为`910B4`，driver/npu-smi版本`25.5.0.b060`。
由观测到的910B家族归类为A2；这是硬件家族归类，不是推断某个Provider兼容。
vLLM-Ascend官方FAQ分别列出910B kernel对应Atlas A2以及单独的Atlas A3支持范围。
[官方设备及SoC说明](https://docs.vllm.ai/projects/ascend/en/v0.18.0/faqs.html)

| 硬件范围 | 本轮观测 | Provider验收 |
|---|---|---|
| A2 / 910B4 / 124卡2、5 | 已实际读取型号；精确整机SKU未读取 | 仅沿用具体CANN/模型/算子/构建组合的真实回执 |
| A3 | 未观测到A3设备 | `UNVERIFIED`，没有A3构建或运行证书 |
| 其他A2 SoC或其他机器 | 无本轮实际设备记录 | 不由同属A2推断兼容 |

## 具体CANN安装及历史验收

以下安装来自先前独立镜像内的安装元数据、头文件、导出符号及编译探针，**不是宿主同时安装四个版本**。
当前宿主只读盘点只列出9.0路径。该盘点阶段没有serving Worker加载证明；随后本轮临时Worker在8.5.1镜像检查了20/39匹配preload和真实RPC，QMM TP2另观测到真实调用，见[本轮Worker回执](../experiments/overnight_20261116/worker_receipts-a2.json)。这些结果不授予新故障注入资格，也不扩展到9.x。

| 具体版本 | 内部安装版本 | 历史Fault39结果 | 当前统一入口结论 |
|---|---|---|---|
| 8.5.1 | V100R001C25SPC002B220 | 算子B/F/OFF；Qwen3 TP1/TP2 | 旧包范围不自动授予新Worker/其他Provider |
| 9.0.0 | V100R001C10SPC001B250 | 算子B/F/OFF | 完整模型未验收 |
| 9.1.0 | V100R001C11SPC001B243 | 算子B/F/OFF；Qwen3 TP2 | 同上，需单独确认当前Worker实际加载 |
| 9.2.0 | V100R001C12SPC001B237 | 算子B/F/OFF | 完整模型未验收 |
| 8.0/8.1/8.2/8.3/8.4及其他定制/RC | 本轮盘点范围内无对应实际profile | `UNVERIFIED` | 不按8.x/9.x字符串放行 |

各profile记录原始CANN库及独立SONAME副本的完整SHA256、可执行段SHA、头文件集合、
必需cache/metadata/Matmul导出以及ABI事实。本轮另外用4个专用device-free CPU容器只读检查
`libopapi/libopapi_nn/libnnopbase/libascendcl`，16份文件SHA均匹配此前精确profile，
实际镜像ID、DT_NEEDED和SONAME见[environment_elf-a1.json](../experiments/overnight_20261116/environment_elf-a1.json)。
这些库的`readelf -n`成功返回但没有GNU Build ID，保留null并记录原notes输出SHA；不从版本或文件SHA伪造Build ID。
这些新容器没有设备映射，未加载CANN库、未导入torch-npu，结束后均已移除；不是NPU调用验收。
如果当前Worker映射文件可读，现有环境接口会解析其GNU Build ID；必须保持映射inode、设备号与文件身份一致。

## 自动发现与拒绝规则

[environment.py](../fault_injection/environment.py)继续不导入或初始化NPU：

- `discover_installations()`按实际manifest枚举当前namespace内的安装，保留未知、缺失及读取失败。
  默认只读元数据和少量ABI头文件；离线盘点可显式开启库SHA/ELF Build ID，避免每次Worker RPC读取全部未用库。
- `inspect_environment()`记录包的installed/loaded版本差别、当前toolkit、映射库和driver。
  只有torch-npu已加载且`is_initialized()`明确为true时才查询实际device name；不会为了inspect初始化设备。
- `loaded_installation_links()`只把经inode/设备号核对的实际映射文件与安装manifest关联。
  `/native`代理库及无法关联的文件保留unresolved，多安装共存保留`MULTIPLE_INSTALLATIONS_LOADED`。
  这仍不是私有ABI/tiling兼容证明。
- `classify_soc()`识别已观测SoC名；用户填写“A3”不构成设备观测，不会被转换为A3资格。
  各Provider还必须核对实际SoC、API、头文件、构建、加载身份和调用边界。

Fault39四版本签名事实与差异说明沿用[精确版本适配文档](../fault39/docs/cann_version_adaptation.md)：
公共Matmul签名相同不证明private cache布局相同；AddRmsNorm的const-output声明发生差异；
FIA V1/V2/V3必须使用对应typed wrapper；26实际依赖自定义AddRmsNormBias，不能别名到标准AddRmsNorm。
20、26、共享量化及33各自固定的CANN/vendor范围不随Fault39版本适配而扩大。

本轮没有下载或覆盖vendor库。矩阵中的历史NPU结果明确标记重用范围，
新的CPU/模型结构测试不升级成`NATIVE_VERIFIED`或`MODEL_CLOSED`。
