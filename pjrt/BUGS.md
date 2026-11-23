# PJRT 错误与复现记录

每个条目记录触发条件、实测结果、修复和验证方法。除非另有说明，
真机是 vivo Y36m（Mali-G57 MC2，Android 13，GPU 驱动 Vulkan 1.1.177）。

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
