# r8 开发安装检查（不是发行包）

124 A2主机，现有0.26/CANN9.1镜像；无设备挂载、network none、不创建Worker、不初始化NPU。三wheel安装后395个源文件逐字节一致，dcatpfi-product fault --help退出0，发行目录直接存在docs。

来源83b30d90，构建位于私有.codex-tmp/r8-validation-build-a1。为验证打包逻辑暂沿用开发树现有版本号，不上传wheel，不把它当作r7.1原包，不修改任何冻结包/标签。正式r8冻结前必须升级三wheel版本并重新执行完整发行门禁。
