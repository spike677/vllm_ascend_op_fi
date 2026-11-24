# 41/26fia/20：构建身份文本回传

最初的`NATIVE_BUILD_41_26FIA_20_OK`只有结论。后续ff19425f文本已补充：Qwen镜像v0.26.0rc1-a3、本地镜像Id、空RepoDigests、SDK文件9.1.0、recipe名称、库29088字节与SHA、回执9057字节与SHA、十条构建/CPU命令成功和四个导出。11-23下午又声明源码r5/d4ae552d和g++11.4.0，但只给源文件SHA前缀及extract SHA。当前仍是**构建未审查**：缺完整提取JSON，不能核验逐源SHA和字段一致性。不要把该行或SDK文件版本转换为permit或已加载运行时观测。

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

## 11-23 下午更新

已收到声明：源码r5/d4ae552d、g++11.4.0、10条命令成功、五个源文件SHA前缀，提取SHA为`5f863898a877f6dd0f181949e66cfcce07e3a7673fba5632a594dc61d21863d2`。**尚未收到完整提取JSON**，因此不能验证其自校验hash或将前缀补成完整源SHA。只需负责人转交完整原JSON，不重新构建，不改变已有字段。

本地已完成[r5/r6源码逐字节对照](native_r5_r6_source_review_20261123.json)：五个Native C++/头文件、recipe及README全部相同，完整期望SHA已列出；这些期望值不能代填黄区观测值。后续验收拟用r6独立包及它的许可SHA，审查通过的同源码库可作为候选，无需仅因包名变化重建。r6共享Python模块有变化，不能继承r5资格；产品模式迁移放入r7。

完整文本到达后按顺序核对：规范化JSON的extract_sha256；source_files与r5逐项一致；SDK版本文件SHA `a9b17ef608ebeb70ade09cde1c12a545eb5e8f3e5312f6806ca0d368b0464848`及9.1.0；g++11.4.0；十条命令及返回码；库/receipt实际字节SHA与既有文本一致；source_release/commit与recipe相符。保留`DEPLOYMENT_FIELD_MISSING:cann_runtime`作为“运行时待核对”，不是SDK构建失败。当前工具输出不包含完整nm/readelf内容，若此处有疑点要求补对应只读文本，不从库名称推断依赖。

完整审查通过后再发布41→26fia→20的A3验收步骤，先TP2后TP1：native_library的真实绝对路径与文件SHA由可信部署装入binding，不能让公共请求指定库或裸地址；库SHA参与运行来源核对，包的source_manifest_sha256仍取r6 source-review.json，二者不能互换。真实消费者、同stream顺序、KV/residual后续合法更新的恢复边界、每rank两作用域与负向检查分别验收。cann_runtime在这次正常验收Worker中读取并记录，不为其单独创建Worker。全文未到之前不发执行许可，不进行三项注入或产品接线。
