# 41/26fia/20：构建身份文本回传

最初的`NATIVE_BUILD_41_26FIA_20_OK`只有结论。后续ff19425f文本已补充：Qwen镜像v0.26.0rc1-a3、本地镜像Id、空RepoDigests、SDK文件9.1.0、recipe名称、库29088字节与SHA、回执9057字节与SHA、十条构建/CPU命令成功和四个导出。当前仍是**构建未审查**：缺实际构建源码发行版/提交、逐源文件SHA、编译器版本及工具提取文本。不要把该行或SDK文件版本转换为permit或已加载运行时观测。

对现有receipt.json和库提取，**不重新构建**。完整已知SHA和runtime-identity.json模板见[统一工具说明](yellow_tools_supplement_20261123.md)。image_id填已回传的完整1a9d274a…本地Id，repo_digests=[]，cann_runtime=null并说明无设备构建未观测运行时；不是填9.1.0推断值。

要求的构建组合是Qwen专用v0.26.0rc1-a3/CANN9.1.0，recipe为发行源码中的`tools/build_native_regions.py`，源目录`fault_injection/native_adapters/postwrite`。旧fault26.fia.abi0.v1/fault20.qmm.abi0.v1或CANN9.2.1构建物不适用。

回传：镜像Id和RepoDigests、实际已加载CANN版本与读取来源、SDK版本文件及SHA、recipe/源文件SHA、编译器、各构建/替身测试返回码、库文件名/字节数/SHA、receipt.json文件名/字节数/SHA。运行容器镜像检查按[镜像声明说明](yellow_image_attestation_20261123.md)的只读命令，不能填A2默认值。无设备构建没有运行时CANN观测时保持null并说明，后续从专用实例补读，不为检查启动Worker。

已有构建脚本会生成receipt.json。本次独立工具补充件新增只读提取命令：

```bash
PYTHONPATH="$TOOLS" python -m tools.extract_native_build_record \
  /private/build/receipt.json /private/build/libdcat_postwrite.so \
  --deployment-identity /private/build/runtime-identity.json
```

runtime-identity.json由实际观测填写`image_id`、`repo_digests`、`cann_runtime`、`cann_runtime_source`；没有的字段null。命令只读、只输出≤4096字节JSON，缺失写issues、库字节与回执不符明确标出，不截断日志冒充通过。输出自带extract_sha256；蓝区仍须将source_files与相应发行源码逐项核对，检查SDK/依赖/导出；该提取本身不发放资格。若旧构建根本没有该schema回执，返回WRONG_NATIVE_RECIPE_RECEIPT并停止，不包装旧产物。

镜像/SDK/运行身份与源码都审查通过以后，才继续41、26fia、20产品模式接线及A3验收。三机制要分别确认消费者、同stream、恢复点和两个作用域；某一条构建通过不代表三条注入均已验收。
