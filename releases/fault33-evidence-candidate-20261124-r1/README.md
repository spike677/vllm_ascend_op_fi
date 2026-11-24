# 33 A3只读取证补充件 r1

只读候选工具，不是r7.5注入发行，不改变r7.4或r7.4.1产品准入。
来源和逐文件SHA见manifest.json；20项CPU通过不能代替A3实测。

使用Python 3.12、与运行实例一致的torch/torch_npu和CANN 9.1 SDK。
在新的私有工作目录解压r7.4.1完整源码包，再覆盖解压本补充件ZIP；不要覆盖现役安装或冻结发行目录。
核对manifest全部文件SHA以及base_dependency SHA。进入该新目录，将src、仓库根、fault39/src和experiments/worker_core_v11_20261118加入PYTHONPATH。
按docs/worker_core_v1/fault33_candidate_windows_20261124.md执行两个窗口：完整捕获与字段审查先通过，再运行控制重算；由负责人转交。
公开候选锁定CANN ops-transformer v9.1.0；实际A3边界、三组形状、每组每rank >=99%和完整差异位置缺一不可。任一不通过就停，不能套用A2的S2参数。

CPU复核：
```bash
PYTHONPATH=src:.:fault39/src:tests/worker_extension python -m unittest test_fault33_candidate_decode test_fault33_candidate_partitions test_paged33_difference_locations test_paged33_per_head_control test_paged33_private_capture -v
```
私有Q/K/V与逐元素差异留在当地，不上传Git。原件完整性、三组调用对应关系和控制结论按已有文本协议回传。本补充件不新增或授予注入组合。
