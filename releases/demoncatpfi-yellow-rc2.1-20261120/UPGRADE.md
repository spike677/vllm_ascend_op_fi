# 黄区 master：RC1.1 → RC2.1

由负责人转交 openchamber；蓝区没有向黄区发送或执行升级。仅升级专用测试部署，不操作对外服务。代码接线继承RC1.1公开接口，不给runner补字段、不改准入。

1. 固定标签 `demoncatpfi-yellow-rc2.1-20261120`，核对其源码提交 `8c358961d9db16bd2bcbf6916bd8d9f5722d3a18` 和本目录 SHA256SUMS。不要合并开发线、TP2或RC2.2。
2. 审查源码变更：相对de17267b，执行源码只变 `reviewed_deployments.py`；发行提交另含本版制品与说明。
3. 准备新空目录，以冻结环境对应Python离线安装下列三个wheel。版本号与RC2相同，必须用本目录SHA校验后的文件，不复用旧安装目标。
4. 在无模型导入的新进程执行源包内 `tools/check_rc2_install.py`；它检查导入来源、版本和默认关闭边界，不运行模型。确认285个安装源码文件与 install-summary.json 的安装文件SHA一致。
5. 现有可信宿主配置 `YELLOW_TEST_DEPLOYMENT` 并调用已审查的公开binding安装/移除接口；这三个精确登记组合不再依赖临时验收许可。未登记的组合继续拒绝。
6. master直接从RC1.1采用此发行，不先上线未登记RC2。保留旧部署以供回退；失败时保留回执并回退旧完整部署，不混装新旧模块。
7. 回传升级版本、安装校验和实际调用结果的文本；不据安装成功新增现役/TP2验收结论。

```sh
# 在下载的本发行目录中；TARGET 必须是新空目录。
sha256sum -c SHA256SUMS
python -m pip install --no-index --no-deps --no-cache-dir --target "$TARGET"   wheels/demoncatpfi-0.3.0-py3-none-any.whl   wheels/demoncat_worker_extension-0.2.0-py3-none-any.whl   wheels/demoncat_fault39_core-0.2.0-py3-none-any.whl
# SOURCE 为已校验源码包中 demoncatpfi-integration/sources。
PYTHONPATH="$TARGET" python -B "$SOURCE/tools/check_rc2_install.py" --expected-root "$TARGET"
```

CANN登记取运行读取器的8.5.1，不修改toolkit文件的8.5.0；不把此A3小模型TP1登记写成现役模型。22的行级summary缺口在RC2.2修复，本版不改总结器。
