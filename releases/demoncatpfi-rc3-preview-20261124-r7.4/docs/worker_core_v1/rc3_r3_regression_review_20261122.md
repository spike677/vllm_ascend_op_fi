# RC3 r3 测试欠账子集复核

范围是本次六项能力、共享 Runtime/准入/多 rank/请求计划/租约/消费者/发行工具，含旧 39 与 Batch 机制回归。原始 197 个 REAL_DEFECT 失败出现次数中，85 项进入本次子集并复测，112 项 KV41/Writer26 Native 研究协议欠账保留，待 Native 阶段逐项处理；不把排除项标成通过。逐项映射和全部执行模块见 `rc3_r3_cpu_gate_20261122.json`。

主要门禁执行 971 项：0 失败、0 错误、2 项原有弃用 skip，约 189.77 秒。补充执行 33 项：0 失败、0 错误、无 skip，约 23.23 秒。两组存在重叠，不称为 1004 个独立测试。CPU 测试不替代 NPU 实机或 A3 验收。

两个 skip 是旧 attention/S2 通过公共工厂直接进入研究模式的用例；对应公开入口拒绝由 ordinary_admission 等负向测试覆盖。未新增 skip。原始分类记录不覆写，新的逐项结论另存。

修正内容：旧 39/Batch 六阶段研究测试明确使用仅 CPU 的 research_fixture，保留消费者、所有权、失败恢复断言；普通 39 四阶段测试继续走真实公共 registry，不用研究 resolver。公共 39/33 不再期待旧工厂回退。25 的未知版本也返回明确“不支持”，同时保留 prerequisites 结构化诊断，研究入口仍拒绝。

扩展首轮 971 项出现 46 失败、1 错误：研究夹具影响普通39测试、25号异常未携带 prerequisites。已修复并完整重跑。宿主 next_phase 循环另用实际多 rank CPU Runtime 验证 world_size=1/2/8、同组两个作用域；不硬编码四次循环。

复跑方式：从 JSON 的 gate.modules 和 additional.modules 分别用 unittest 加载执行，PYTHONPATH 采用绝对路径的 repo、repo/src、repo/tests、repo/tests/worker_extension、repo/fault39/src，避免子进程目录变化导致环境假失败。本次是声明范围内的发行门禁，不是全仓所有历史研究路线全绿。
