# 固定FakeRequant参考依赖

- 来源：`research/fake-requant-reference` 的提交 `1d715725f78a64ffb56bf8d1d75ec37275aef51f`，通过git archive取得，不读其未提交算法。
- [运行源码](source/fake_requant_reference_extension.py) SHA256：`56097835861bb441232663380eed2953acd97f3aa03c49262368dce6defeaca7`。
- [原数学/config测试](tests/test_fake_requant_math.py)按同提交原样纳入。

Linux显式将source加入PYTHONPATH；当前演示启动器要求将此文件复制到新session/reference并核对SHA。combined.py严格校验环境变量ASCEND_FI_REFERENCE_SHA256与实际加载文件；不自动下载或回退其他工作树。

输入是匹配模块的浮点tensor和config，输出是原dtype替换输入；IDENTITY不做Q/DQ。唯一[算法说明](../../docs/architecture/algorithms_zh.md)记录eps、计数、清理边界及CPU helper极小scale差异，本轮不改数学。
