# r8 可信 bootstrap 与登记门禁：A2 实测

组合：Ascend910B4/A2、vLLM0.26/CANN9.1、社区Qwen3.8 W8A8、TP2。r8开发候选，未冻结，今晚不下发。

实际使用官方 demoncat_product_bootstrap 的 configure/permit 路径（本地可信初始化保留部署鉴权回调），18.api_half 在同组Worker两个作用域完成真实HTTP产品验收。独立 review_product_mode 通过，窗口282.309秒，隔离清理器已停止。记录 review_18.json 与 window_r8-registration-acceptance-a2.json。临时验收许可存在时的通过，不是免许可通过。

登记身份：两rank实际implementation SHA均为1bea0c5f9d43f2295d5dcce476681f7b4eba5d7bd8e8651efbe52c26eeca23b5，与本地开发源码相同；没有残留旧Python文件。4页产品提取通过，但更严格的正式登记门禁拒绝 PRODUCT_REGISTRATION_IDENTITY_MISSING:runtime_libraries。没有创建登记、没有运行伪造正例、没有授予黄区资格。

不映射设备的隔离检查进一步发现：同一容器CANN9.1库的 /proc/self/maps 设备号与路径fstat设备号不一致，inode相同；map_files描述符因权限拒绝不能打开。因此 environment.inspect_environment 返回 MAPPED_FILE_IDENTITY_CHANGED，不能取得 MAPPED_FILE_VERIFIED 的SHA和安装链接。现有Native库自身的审查与此不是同一结论。路径文件的SHA、容器外的SHA、安装版本文件都不能冒充已加载库证明。

下一项需补的是可证明的运行库映射身份适配，并用新实际身份重新验证；当前保持拒绝，不将路径哈希降格成已验证映射，不改旧Native准入。免许可表仍为空，r8无许可两owner正例待完成。身份门禁与CLI helper的CPU负例通过，不替代这个实机缺口。

本轮实际负例仍为缺许可、未知profile、用户Native路径；缺许可路由在卸载binding后也有ACTUAL_CALL_NOT_YET_OBSERVED，不能表述为只差许可的纯负例。开发harness另补Worker侧未知机制99检查，但未重跑NPU，不写成实机通过。黄区真实CLI和daemon不在蓝区，此处也不替其验收。

r8无设备安装的395文件一致及参考console检查见 ../r8_no_device_install_20261123；此前CPU全量1200项（1198通过、2项带理由跳过），随后登记/CLI专项22项和文档3项通过。发行版本号暂未升级；私有验证构建不能作为正式r8包发布。
