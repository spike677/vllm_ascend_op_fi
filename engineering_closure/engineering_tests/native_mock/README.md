# Native 租约测试

`postwrite/postwrite_lease_mock.cpp` 将生产 `fault_injection/native_adapters/postwrite/postwrite.cpp` 直接链接到 CPU 实现的 ACL API。测试包含多个存活区间、重叠拒绝、owner/generation/thread/device 绑定、陈旧 handle、地址复用、部分 memcpy 失败和恢复重试。它验证内存与 ABI 控制逻辑，Ascend 驱动行为需要目标机实测。

在 Visual Studio x64 Native Tools 环境、仓库根目录运行，输出目录须预先创建。

```bat
cl /nologo /std:c++17 /EHsc /W4 /WX /I fault_injection\native_adapters\postwrite /I engineering_closure\engineering_tests\native_mock\postwrite /Fo:.codex-tmp\postwrite-lease-mock\ /Fe:.codex-tmp\postwrite-lease-mock\lease.exe fault_injection\native_adapters\postwrite\postwrite.cpp engineering_closure\engineering_tests\native_mock\postwrite\postwrite_lease_mock.cpp
.codex-tmp\postwrite-lease-mock\lease.exe
```

Linux 可用 C++17 编译器，以相同两个 include 目录和源码构建并增加 `-pthread`。动态库加载测试单独执行 `python3 -m unittest discover -s engineering_closure/engineering_tests/linux_loader -v`，需要 C 编译器。
