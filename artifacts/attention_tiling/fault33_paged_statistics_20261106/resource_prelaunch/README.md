# 模型启动前的资源回执

四份 JSON 为远端 `outputs/manual_launch/` 的原始文件，复制后逐项核对远端 SHA，未改写。它们记录控制器启动前的独立辅助步骤，不属于控制器持锁执行窗口。

- B：辅助程序发现并严格核对一个只映射卡 5 的 ATK 容器，完整 ID 为 `f6e170e337056765978975dd78f5f59f9b607823243dd2166f7cf4ddbc8f9dd5`，按用户授权停止成功；没有删除数据或修改队列文件。
- F0、F1、恢复：启动前辅助步骤的 `cleanup=null`，没有执行停止操作。
- 四个正式回合各自通过控制器原有资源锁、重复门禁和自身容器清理。对应 completion 在 `../model_a4/audit/model/attempts/`。

整个实验较早阶段也曾在独立预启动步骤停止获授权的卡 5 ATK 占用；中断 attempt 的部分结果不计入正式验收。原失败回执保留远端，未改成成功状态。没有停止 gyl_test 服务、占用卡 2，或伪造队列完成结果。

辅助程序源码见 `experiments/fault33_paged_tiling_20261105/analysis/launch_model_case.py`，四份回执绑定同一原字节 SHA `068342cb25638230ac6fe79fdfe1a0b6508824fa05c3abe566d7f7531855b193`。控制器的 `external_container_mutations=false` 只说明它自己的执行窗口；不代表整个实验没有发生以上授权清理。
