# 33完整捕获增量：尚未授予A3资格

新增文件 `experiments/worker_core_v11_20261118/paged33_launch_capture_v2.cpp`，不覆盖旧采集器。公开ACL函数声明在124已有A2 CANN9.1镜像内无设备编译通过；不代表A3已加载库、serializer或分组已匹配。

独立CPU模拟：9000个不同函数handle、9000次原生转发，全部捕获；16条容量限制正确报告8984条capacity_drop；过大host参数报告9000条oversize；未开启窗口报告9000条unarmed。fake ACL检查原参数和返回值，未调用NPU。原件统计见[r74_capture_cpu_20261124.json](r74_capture_cpu_20261124.json)。本地审查器另测试8193条、文件截断、缺manifest、计数矛盾与未知handle。

## 有界采集配置

可信部署方在专用实例中编译和使用；先核对A3实际加载的CANN库ABI、源码和构建身份。库不调用设备读写，不改launch参数，不新增kernel。

```bash
c++ -std=c++17 -shared -fPIC \
  -I "$ASCEND_HOME_PATH/include" \
  experiments/worker_core_v11_20261118/paged33_launch_capture_v2.cpp \
  -ldl -pthread -o capture-v2.so
```

每轮使用新的输出目录；预加载到实际Worker，保留部署已有LD_PRELOAD链。配置：

```bash
export DCAT33_CAPTURE_DIR=/path/to/new-window
export DCAT33_CAPTURE_ALL=1
export DCAT33_MAX_RECORDS=65536
```

初次完整短窗口用ALL=1，不默认套A2的key。已观测并审核实际key后，可不设置ALL，改用 `DCAT33_CAPTURE_KEYS=实际十进制key,另一实际key`；此时结论仅覆盖选定key。handle表最多131072项、host参数65536字节、placeholder最多128项、进程内累计保留原始文件预算1GiB。达到任一上限会记错误，不能把截断结果算完整。

部署方在一次受控请求前创建输出目录内的空文件 `ARMED`，请求完成并确认各rank本次控制边界到达后移除；正常退出专用Worker，使每进程输出最终 `capture-<pid>.json`。仅在控制边界操作，不在执行kernel的条件分支插入collective。进程被kill或未正常退出导致manifest缺失，判不完整。目录中的二进制host参数可能含地址，原件只保留黄区，不提交Git。

```bash
python tools/review_launch_capture_v2.py /path/to/new-window \
  --pid 实际Worker进程号 --output capture-review.json
```

审查器逐条检查header、完整长度、原生返回码、唯一serial与文件数，核对seen、selected、filtered、written及所有错误计数。unknown/map_drop也拦截，不能假设未知handle不属于FIA。只有完整的选定窗口才返回exit0；`qualification_granted` 和 `serializer_validated` 始终为false。没有任何记录也不通过。计数不一致、写入失败、碰到容量上限、磁盘错误均须换短窗口重采；不补造raw。

## 仍需负责人协调的A3材料

实际Worker映射的operator/host库SHA及build-id、CANN来源、匹配发行源码manifest、serializer定义及sparseMode=3分支/key/模板。现有7928条/最大serial8191不能补全这些材料。新采集器解决可观测的截断和写入漏报，不替代源码匹配；完整目标控制重算逐位一致率≥99%后才可做两个owner注入验收。33不按时限结案，材料缺失继续记“需退补”。

给负责人转交，Codex不直接联系黄区。本文件是增量开发说明，不是r7.4冻结执行单。
