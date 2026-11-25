# PJRT 错误与复现记录

每个条目记录触发条件、实测结果、修复和验证方法。除非另有说明，
真机是 vivo Y36m（Mali-G57 MC2，Android 13，GPU 驱动 Vulkan 1.1.177）。
完成一个 part 时，把本轮复现的问题和验证结果随代码一起提交。

## LLDB 连上手机 Python 后，PJRT 插件断点仍未解析

- **触发**：通过 glibc loader 启动手机上的 Python，经 Android `lldb-server` 附加，在 `loaded_execute()` 设断点。
- **实测**：JAX 能在 Mali 上实际执行，LLDB 模块列表却没有 `libpjrt_mali.so`，源代码断点不触发。
- **修复**：VS Code 调试任务在初始化 PJRT 前预加载插件并记录手机上的映射基址。CodeLLDB 加载同一次构建的本地符号，并应用该地址。
- **回归**：使用 `Mali phone: PJRT plugin in Python` 启动调试，在 `executable.c` 的 `loaded_execute()` 函数体打源代码断点。

## IREE v3.11.0 的 Vulkan 调用会等待设备侧 fence

- **触发**：在 vivo Y36m 上用 `--iree-execution-model=async-external` 编译两段 StableHLO，传入尚未触发的起始 fence。
- **实测**：保持起始 fence 不触发时，探针超过 30 秒未完成。改用 1 秒看门狗后，第一段调用直到看门狗触发才返回。一次 Release 测量为第一段提交 `1010.454 ms`、第二段 `2.398 ms`、最终等待 `0.003 ms`，最后读回 `[5, 7, 9, 11]`。第一段耗时包含人为设置的等待。
- **原因**：该版本 `queue_alloca` 在主机端等输入 semaphore，而 `queue_execute` 在释放转换后的命令缓冲区之前等完成 semaphore。编译器生成 `coarse-fences` 入口后，Vulkan HAL 仍包含这些等待。[IREE v3.11.0 Vulkan 实现](https://github.com/iree-org/iree/blob/v3.11.0/runtime/src/iree/hal/drivers/vulkan/vulkan_device.cc#L1661)
- **复现**：`run_on_phone.sh --fences` 独立编译并运行探针。退出码 `3` 表示数值正确且提交被看门狗释放。接入 PJRT event 前需先让目标 Vulkan runtime 的 fence 探针返回 `0`。

## 同步执行会卡住 JIT 提交

- **触发**：编译并上传后用 `SIGSTOP` 暂停测试 worker，再提交两次有依赖的执行。
- **原因**：旧 `Execute` 持有 socket 锁等整次执行回包，只有回包后才能构造输出和 ready event。
- **修复**：PJRT 预建输出和 pending event，由后台线程按队列顺序运行。任务持有远端资源引用，完成后通知等待者。回调由独立线程执行，可以等待后续任务。
- **回归**：`--async` 验证暂停期间照常提交、event 未完成和提前删除输入与 executable；恢复后验证输出和回调。`--disconnect` 验证暂停期间 JAX 能串两次编译好的调用，worker 退出后两个结果都报执行错误。

## Worker 退出时，JAX 错误缺少失败操作

- **触发**：在手机上完成 JIT 编译和输入上传后终止本次测试启动的 Mali worker，再执行 JIT。
- **实测**：旧版本只返回 `UNAVAILABLE: Lost connection to Mali worker`，无法从消息判断失败发生在执行还是数据传输。
- **修复**：错误包含当前 worker 操作，并区分发送失败、接收失败、无效回复和连接已关闭。
- **回归**：`run_on_phone.sh --disconnect` 等待 worker 退出，再验证等待执行结果和读取原输入都返回带操作名的 `UNAVAILABLE`。

## JAX 0.6.1 的自动磁盘缓存跳过 Mali 插件

- **触发**：为注册名为 `mali` 的后端配置 `jax_compilation_cache_dir`。
- **原因**：该版本的自动缓存入口只接受 `tpu`、`gpu`、`cpu` 和
  `neuron`。实现 PJRT 序列化回调后，仍需显式保存和加载。
- **处理**：使用 JAX 的 `serialize_executable()` 和
  `deserialize_executable()`。序列化镜像携带 VMFB 校验值和驱动标签。
- **回归**：`run_on_phone.sh --cache` 用两个独立进程验证。第二个进程关闭
  编译入口，拒绝错误头部、损坏的 VMFB 和不匹配的驱动，再用有效镜像跑两次训练。

## G57 上通用 Vulkan 目标产生错误的 FP16 结果

- **触发**：用 IREE v3.11 的通用 Vulkan 目标编译 `x + x`，输入
  `float16[[0.5, 1.25]]`。
- **实测**：独立 Android IREE runner 返回 `[[0.5, 1.375]]`。
  同一份输入用 `--iree-vulkan-target=mali-g57` 编译后返回 `[[1, 2.5]]`。
- **修复**：PJRT 编译时为 Mali-G57 显式选择 `mali-g57` 目标。
- **回归**：`run_on_phone.sh` 检查二维和奇数长度的一维 FP16 数组。

## 输出声明会把无符号类型写成 signless 整数

- **触发**：编译返回 `uint32` 的函数，然后检查 VMFB 的
  `iree.abi.declaration`。
- **实测**：底层签名包含 `tensor<...xi32>`，逻辑类型保存在
  `iree.abi.encoding = tensor<...xui32>`。
- **修复**：从 ABI 编码取逻辑 dtype，并校验执行返回的形状和存储表示。
- **回归**：`run_on_phone.sh --abi` 在执行前查询 `uint32[2,3]` 的元数据，
  常规测试验证带最高位的无符号整数实际运算。

## 默认 XLA debug options 的 protobuf map 顺序不稳定

- **触发**：按序比较两次序列化的默认 XLA debug options。
- **实测**：在手机上首次 `jax.jit` 被误报为自定义 debug 选项。不同的是
  字段 357 的 map 条目顺序。
- **修复**：对 protobuf map 比较条目及重复次数，普通字段仍按序比较。
- **回归**：常规 JIT 通过，C ABI 测试仍拒绝自定义 debug 选项。

## 读取优化后 HLO 时触发空 PJRT 回调

- **触发**：首次运行 JAX JIT，JAX 查询 executable 的布局或 HLO 信息。
- **实测**：LLDB 停在由空回调引发的 `SIGSEGV`。发生在编译之后、首次执行之前。
- **修复**：实现 `PJRT_Executable_OptimizedProgram`，对 IREE 后端没有的优化后
  XLA HLO 返回 `UNIMPLEMENTED`，让 JAX 走其回退路径。
- **回归**：JIT 加法、多输出和连续训练均通过。

## Android 文件路径形式的 Unix socket 无法绑定

- **触发**：以 ADB shell 身份在 `/data/local/tmp` 下绑定 socket 文件。
- **实测**：worker 返回 `bind: Permission denied`。
- **修复**：使用 abstract Unix socket。测试为每次运行生成唯一地址。
- **回归**：设备枚举、数组往返及缺失 worker 错误测试均通过。

## Vulkan loader 和物理设备报告的版本不同

- **触发**：根据系统 loader 的 Vulkan 版本判断当前 IREE runtime 能否使用。
- **实测**：loader 是 1.3.0，Mali-G57 物理设备是 1.1.177。较新的 IREE
  Vulkan 后端要求物理设备达到 Vulkan 1.3。
- **处理**：这台手机的编译器和 Android runtime 固定为已验证的 IREE v3.11.0。
- **复现**：运行根目录 README 中的 Vulkan 探针。
