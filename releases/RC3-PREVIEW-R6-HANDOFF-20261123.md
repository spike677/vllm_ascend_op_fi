# RC3 预览 r6 冻结交付（2026-11-23）

标签：`demoncatpfi-rc3-preview-20261123-r6`。源码提交：`0e5d3e34650df2c165509fe2d3180c8c3e53a8e7`。本次为候选包冻结，不是A3登记发行，product_closed=false。r5及其独立提取补充件保持原字节，r5.1仍只用于后续黄区登记。

## 包与许可身份

- [完整发行目录](demoncatpfi-rc3-preview-20261123-r6/)，包括三wheel、源码tar、源码/发行清单、校验清单、CPU回归与安装回执。
- demoncatpfi：`0.4.0rc5`；demoncat-worker-extension、demoncat-fault39-core：`0.3.0rc5`。版本号与预览序号不同，不混用r5 wheel。
- 许可的`source_manifest_sha256`填写 **source-review.json原始字节SHA256**：

```text
7a411ce29de51a0fecc6dbfe850aeea765f948c6414708360c873dfe22f38bf4
```

它不是release-manifest.json的SHA，也不是实机旧运行目录的manifest。源码tar SHA为`5d3a173df1737278e3eb20a6c846397ce3646d1ed4e888eb96fe1aec3dfd39a6`，完整文件SHA见[SHA256SUMS](demoncatpfi-rc3-preview-20261123-r6/SHA256SUMS)。

## 本次纳入与保持关闭

产品模式候选共八项：原18、22、39op、28、17，以及新增Batch、39mr、33。新增三项此前在A2/0.26/CANN9.1/Qwen3.8社区W8A8/TP2、同组Worker两个作用域完成真实HTTP请求自动发现、全rank符合条件调用持续触发、真实消费者、区域恢复/资源零/关闭后金丝雀精确恢复。运行时原清单SHA各自保留在[独立结果](../docs/worker_core_v1/product_r6_20261123/README.md)，不声称本次新wheel又做了一轮NPU重跑。

- Batch：max_num_seqs=8，实际decode B2～8，B1不注入；共享调用按rank/frame去重。不是所有HTTP并发都保证同批。
- 39mr：保持完整K控制与错误范围证据，范围外调用明确拒绝。
- 33：只验收A2的2304-token prompt、4生成token及已测分组。**A3产品入口仍关闭**；提供[取证→原生S2解析→固定QKV控制≥99%→两作用域验收步骤](../docs/worker_core_v1/rc3_r6_paged33_a3_acceptance.md)。当前A3仅可做只观测和控制检查，原生布局/adapter审查完成前不提供强行开启命令；不能直接复制A2分组。TP1同样不能继承TP2资格。
- 41、26fia、20：既有验收模式仍要求匹配且已审查Native构建，产品模式保持关闭。
- 8及DeepSeek新部署：适配/观测/噪声和产品验收未完成，保持关闭。未验收组合不因包内有源码而放行。

候选与限制详见[r6范围](../docs/worker_core_v1/rc3_r6_candidate.md)、[RPC契约](../docs/worker_core_v1/product_rpc_contract.md)和[12项进度表](../docs/worker_core_v1/product_progress_20261123.md)。

## 与r5相同的冻结门禁

- 1103项CPU测试：1101通过、0失败、0错误，2项既有废弃Native研究入口skip及理由逐项保留在regression-review.json。
- 124无设备、无网络独立容器安装：三wheel版本正确，**384个安装源文件逐字节一致**；未创建Worker、未初始化NPU。执行时间2026-11-23 11:27:28～11:27:33（北京时间）。
- 源码tar内1176个文件与source-review逐文件核对通过；包含新产品入口、两个提取工具、冻结r5审查器副本、17图片代码、33取证工具及步骤。
- 新观察器在124 CANN9.1隔离容器无设备编译通过；仅说明编译/接口检查通过，不代表A3 launch布局已验证。A3须在自身镜像重新构建。
- r5冻结目录与原tag逐字节未变。本次没有NPU测试窗口，不停止外部任务，不设置调度隔离。

## 独立安装与核验

仅在匹配软件栈的新安装目录执行，不覆盖现役/原r5目录。RELEASE为本发行目录，SOURCE为新解压源码目录，SITE为独立wheel安装目录；先使用现役匹配镜像提供torch/vLLM等依赖。

```bash
cd "$RELEASE"
sha256sum -c SHA256SUMS
mkdir -p "$SOURCE" "$SITE"
tar -xzf demoncatpfi-rc3-preview-20261123-r6.tar.gz -C "$SOURCE"
python -m pip install --no-index --no-deps --target "$SITE" "$RELEASE"/wheels/*.whl
python "$SOURCE/tools/check_rc3_preview_install.py" --site "$SITE" --release "$RELEASE"
```

安装工具必须passed=true、issues为空；部署仍要核对硬件原名、镜像digest、实际版本、模型config/量化描述/所有权重索引SHA和调度参数。可信部署安装binding与许可；宿主保持attach/enable/status/disable/detach，用户正常发请求。公共请求不能填层名、rank、裸地址或修改数值参数。未审查产品组合保持`PRODUCT_ACCEPTANCE_ONLY`，不宣称主项目产品已关闭。

## 17与提取文本

17无`--image`时使用`vision_input.deterministic_image()`生成同一64×64图，并记录像素SHA；该参数仍可选。产品金丝雀用内置图，用户图仍来自真实请求。A3 TP2必须设置`VLLM_WORKER_MULTIPROC_METHOD=spawn`。历史r5固定图结果仍有效，不要求重跑。

源码含`tools/extract_registration_record.py`和`tools/extract_dsv4_record.py`。r5登记提取器始终加载`tools/_r5/review_continuous_public.py`，SHA为`deb3c5c0d600f7bcae7d0997cfd883591ece87231ed8128e486edb5d823ffadc`，不使用r6审查器代替。十四组合运行及`--verify`沿用[独立工具说明](RC3-PREVIEW-R5-EXTRACT-TOOL-20261123.md)。不要把r6产品回执输入r5登记器后改身份冒充r5。

完整summary和原始回执留黄区，只回传本地提取文本和原件SHA。33阶段1/2按其步骤回传每rank的短JSON/布局解析证据，不回传地址或完整QKV。所有材料由负责人转交，本次未直接向黄区发送，也未新增黄区组合登记。
