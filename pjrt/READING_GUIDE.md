# JAX Mali PJRT 源码阅读指南

这份指南从 C 的调用方视角追踪一次 `jax.jit`。第一遍建议读 `abi_smoke.c` 的 `dispatch()`，接着读 `executable.c` 的 `loaded_execute()`、`async.c` 的 `execute()`、`worker_buffers.c` 的 `execute_module()`。这样可以先弄清一次调用怎样跨过进程边界，再回头看编译、缓存和调试。

## 两个进程和三份接口

```text
vivo Y36m
  Linux arm64 / glibc                      Android arm64 / Bionic
  Python → JAX → libpjrt_mali.so            mali_worker → IREE runtime → Vulkan → Mali
                       │                         │
                       └── abstract Unix socket ┘
                       │
                       └── 手机上的 iree-compile 子进程：StableHLO → VMFB
```

Windows 构建的是可部署的 C 插件和 Android worker。模型的 JIT 编译由手机上的 `iree-compile` 完成。IREE 编译器接收 StableHLO 文本，输出 `.vmfb`，Android worker 加载这个文件并执行。输入、输出和 executable 的远端句柄都归当前 socket 会话管理。

| 接口 | 谁调用谁 | 从哪读 |
|---|---|---|
| PJRT C API | JAX 通过一张函数指针表调用插件。`GetPjrtApi()` 返回表，`PJRT_*_Args` 结构传递参数和结果。 | [plugin.c](plugin.c) · [固定 ABI 的构建脚本](build.py) |
| socket 协议 | C 插件向 Android worker 发送 `MALI_LOAD`、`MALI_EXECUTE`、`MALI_READ` 等请求。包里是整数句柄、元数据或字节。 | [protocol.h](protocol.h) |
| IREE runtime 与 HAL | worker 把句柄查成 `iree_hal_buffer_view_t *` 或 IREE session，再调用 runtime。Vulkan 由 IREE 的 HAL 驱动管理。 | [worker_buffers.c](worker_buffers.c) |

`pjrt_c_api.h` 由 `build.py` 按固定 XLA revision 下载并校验，构建后位于 `build/pjrt-deps/`。读 PJRT 回调时，可以把该头文件和我们自己的 `plugin_internal.h` 并排打开。前者描述 JAX 能调用的 ABI，后者描述插件在内存里怎样实现这些对象。[build.py](build.py) · [plugin_internal.h](plugin_internal.h)

## 跟踪一次 JIT

以下调用可在已配置好手机 Python、编译器和运行中的 worker 的环境里理解。第一次 `twice(x)` 需要编译和加载，第二次输入 shape 和 dtype 相同，可以复用 JAX 已加载的 executable。实际的端到端例子在 `check_device()` 里。[真机测试](smoke.py)

```python
import jax
import numpy as np

device, = jax.devices("mali")
x = jax.device_put(np.array([1, 2], dtype=np.float32), device)
twice = jax.jit(lambda x: x + x)
y = twice(x)
z = twice(y)
z.block_until_ready()
print(np.asarray(z))  # [4. 8.]
```

1. **发现设备。** `initialize()` 注册 `mali`，JAX 加载同目录的 `libpjrt_mali.so`。`client_create()` 连上 worker，验证握手，读取 GPU 名称、驱动版本和队列信息，再启动派发线程和回调线程。Android 侧的对应入口是 `open_vulkan()`、`serve_client()`。 [Python 注册](jax_plugins/mali/__init__.py) · [插件入口](plugin.c) · [worker 入口](worker.c)
2. **上传 `x`。** `client_buffer_from_host()` 检查 dtype、shape 和 strides，必要时先把 host 数据排成连续字节。它发 `MALI_ALLOCATE` 和分块的 `MALI_WRITE`。worker 保存 IREE buffer view，插件只保留句柄和本地元数据。 [上传实现](buffer.c) · [设备 buffer 表](worker_buffers.c)
3. **编译并加载 `twice`。** `client_compile()` 校验 XLA 编译选项，`compile_module()` 启动手机编译器，并对 G57 选用 `--iree-vulkan-target=mali-g57`。接着 `load_image()` 发 `MALI_LOAD`。worker 的 `load_module()` 打开 VMFB、找导出函数 `main`，并从 `iree.abi.declaration` 解析输出元数据。输出 shape 和 dtype 在第一次执行之前已经可用。 [编译与加载](executable.c) · [输出签名解析](output_metadata.c)
4. **提交 `y` 和 `z`。** `loaded_execute()` 用输出元数据先构造 `PJRT_Buffer`，挂上 pending event，保留任务用到的资源，再入队返回。后台 `dispatch_main()` 取出任务，`execute()` 等输入就绪并发 `MALI_EXECUTE`。worker 的 `execute_module()` 调用 IREE，把输出句柄回给插件。`complete_execution()` 发布完成状态并唤醒等待者。`z` 可以先排队，执行时再取 `y` 的实际句柄。 [提交入口](executable.c) · [队列和事件](async.c) · [设备执行](worker_buffers.c)
5. **等待和读回。** `block_until_ready()` 通过 PJRT event 等计算完成。`np.asarray(z)` 走 `download()`，先等 producing event，再拿 socket 锁发 `MALI_READ`，将读回的字节交给主机。 [等待实现](async.c) · [读回实现](buffer.c)

如果在 C 里跟消息，统一从 `buffer.c` 的 `exchange()` 跳到 `worker_buffers.c` 的 `serve_buffers()`。最容易看清的三个分支是 `MALI_LOAD`、`MALI_EXECUTE`、`MALI_READ`。其中 `MALI_EXECUTE` 传的是 executable 和输入 buffer 的句柄，跨进程的中间结果仍用句柄表示。 [socket 客户端](buffer.c) · [worker 请求循环](worker_buffers.c)

## 对象、锁和完成时机

| 对象 | 持有什么 | 何时释放 |
|---|---|---|
| `PJRT_Buffer` | shape、dtype、size 和一个 `mali_resource` 引用。执行刚入队时，输出对象已经存在，远端结果句柄还没回来。 | `delete_resource()` 摘掉公开对象的引用。已排队任务仍可使用自己持有的引用。 |
| `mali_resource` | worker 句柄、引用计数、释放操作和 ready event。buffer 用 `MALI_RELEASE`，已加载 executable 用 `MALI_UNLOAD`。 | 最后一个引用释放时，`release_resource()` 向 worker 发释放请求。 |
| `mali_execution` | 一次排队任务持有的 executable、输入、输出和完成 event。 | 派发线程处理完任务后释放各个引用，再释放 job。 |
| `PJRT_Event` | 是否完成、完成状态、等待条件变量和 OnReady 回调。 | 任务、资源和外部 event 句柄都可以持有引用。最后一个引用释放时销毁。 |
| `executable_image` | VMFB 字节及版本、设备和驱动标签与校验值。 | 独立引用计数管理。重新加载镜像会生成当前 worker 会话的新句柄。 |

读生命周期时，从 `loaded_execute()` 找 `retain_resource()`，到 `dispatch_main()` 找成对的 `release_resource()`。`dispatch_lock` 保护队列以及公开对象的资源指针，`worker_lock` 保护成对的 socket 请求和回复。读回在取得 `worker_lock` 前就等输出 event。待完成的 `OnReady` 回调由独立线程运行，因此回调可以等待后续任务，派发线程仍能继续执行。 [对象与锁](plugin_internal.h) · [资源和回调](async.c) · [镜像](executable_image.c)

当前异步发生在 **Python 调用与后台派发线程之间**。正常执行路径由 worker 逐个等待 IREE 调用并回复，PJRT event 根据回复完成。`--fences` 是单独的设备侧验证。它尝试在起始 fence 未触发时提交两段相依计算。在这台 Y36m 上，固定的 IREE v3.11.0 等到看门狗释放 fence 才从第一段提交返回，最后结果正确，探针退出码是 `3`。`0` 表示两次提交在 fence 触发前返回。 [探针实现](fence_probe.c) · [复现和原因](BUGS.md) · [探针命令](README.md#probe-vulkan-fence-submission)

## 文件索引

### 后端实现

| 文件 | 用它回答的问题 |
|---|---|
| [`jax_plugins/mali/__init__.py`](jax_plugins/mali/__init__.py) | JAX 怎样发现 `mali`，默认编译选项怎样交给 C 插件。 |
| [`plugin.c`](plugin.c) | 哪些回调被注册，设备信息和错误对象怎样构造。先看末尾 `GetPjrtApi()`、`initialize_api()`，再回到 `client_create()`。 |
| [`plugin_internal.h`](plugin_internal.h) | C 插件各对象如何互相引用，队列和锁放在哪。 |
| [`protocol.h`](protocol.h) | 握手包和数据包长什么样，两个进程如何读写 abstract socket。 |
| [`buffer.c`](buffer.c) | 上传、读回、layout、dtype 和删除怎样映射到 PJRT 回调，也包含共用的 `exchange()` 和错误映射。 |
| [`executable.c`](executable.c) | JIT 怎样调用 IREE 编译器，VMFB 怎样加载，执行怎样入队，保存和重载走哪个入口。 |
| [`async.c`](async.c) | 排队、派发、等待、回调和远端资源释放的顺序。 |
| [`compile_options.c`](compile_options.c) | 当前支持怎样的 XLA 请求。先读 `validate_compile_options()`，再追 protobuf 解析。支持单设备、单 replica 和单 partition。 |
| [`executable_image.c`](executable_image.c) · [`executable_image.h`](executable_image.h) | 可执行镜像的磁盘格式、兼容性校验和引用计数。头文件暴露读取和访问接口。 |
| [`worker.c`](worker.c) | 如何打开 Mali Vulkan 设备、接收客户端、握手并进入请求循环。 |
| [`worker_buffers.c`](worker_buffers.c) · [`worker_buffers.h`](worker_buffers.h) | worker 如何维护 IREE buffer 和 executable 表，分派请求。头文件暴露 `serve_buffers()`。 |
| [`output_metadata.c`](output_metadata.c) · [`output_metadata.h`](output_metadata.h) | 从 IREE 导出签名读取静态 shape 和 dtype，用 `iree.abi.encoding` 把底层的 signless 整数恢复成逻辑上的 `uint32`。头文件暴露解析入口。 |

### 构建、验证和说明

| 文件 | 作用 |
|---|---|
| [`pjrt/build.py`](build.py) | 构建并组装手机端 bundle。Zig 产出 Linux arm64 插件及 C ABI 测试，Android NDK 产出 Bionic worker。 |
| [`pjrt/iree_runtime_plugin.cmake`](iree_runtime_plugin.cmake) | 将 `mali_worker` 加进 IREE 的 CMake 构建，列出 Android 侧源文件与链接库。 |
| [`pjrt/run_on_phone.sh`](run_on_phone.sh) | 设置手机端 Python、glibc loader、JAX 平台与编译器路径，再运行 smoke。 |
| [`pjrt/smoke.py`](smoke.py) | 真机 Python 测试和 worker 启动器。覆盖 `jax.device_put()`、JIT、训练、重启缓存、断连和探针。 |
| [`pjrt/abi_smoke.c`](abi_smoke.c) | 直接 `dlopen()` 插件的 C API 示例。`main()` 创建 client，`dispatch()` 展示 `Execute` 参数，`check_async()` 暂停 worker 检查 pending 和提前释放。 |
| [`pjrt/fence_probe.c`](fence_probe.c) · [`pjrt/fence_probe.h`](fence_probe.h) | 两段 StableHLO 导出函数的设备侧 fence 实验和 worker 调用入口。 |
| [`pjrt/README.md`](README.md) · [`pjrt/BUGS.md`](BUGS.md) | 后端构建、复跑、缓存和调试命令，以及每个已复现问题的触发条件、实测结果和验证方法。 |

### 根目录和编辑器

根目录还保留直接调用 `iree-run-module` 的实验。这个入口可以把排查范围缩到编译器、IREE runtime 和 Vulkan。

| 文件 | 作用 |
|---|---|
| [`README.md`](../README.md) | 项目总入口，记录手机环境、初始实验和 Android IREE runtime 的构建。 |
| [`smoke.py`](../smoke.py) | 前向、MSE、单步和八步 SGD 的训练代码，也可以导出 StableHLO。构建 PJRT bundle 时复制为 `training_fixture.py`。 |
| [`train_on_phone.sh`](../train_on_phone.sh) | 直接用手机上的 Python 和 IREE 编译器生成 VMFB，再用 Android runner 执行并核对权重。 |
| [`verify_android.py`](../verify_android.py) | 从电脑用 ADB 推送现成 runner 与 VMFB，检查手机上的前向和一次梯度更新。 |
| [`vulkan_probe.c`](../vulkan_probe.c) · [`CMakeLists.txt`](../CMakeLists.txt) | 枚举 Vulkan loader、物理设备和计算队列。根目录 CMake 构建这个探针。 |
| [`.gitattributes`](../.gitattributes) · [`.gitignore`](../.gitignore) | 固定 shell 脚本的 LF 换行，忽略构建产物、虚拟环境和 Python 缓存。 |
| [`.vscode/launch.json`](../.vscode/launch.json) · [`.vscode/tasks.json`](../.vscode/tasks.json) | VS Code 启动配置和真机前置任务，分别覆盖本机 CPU、手机 Python 插件、Android worker。 |
| [`.vscode/phone_debug.py`](../.vscode/phone_debug.py) | 构建并推送带符号的包，启动远端调试进程，转发 ADB 端口。调 Python 插件时还记录库的映射基址供 CodeLLDB 加载符号。 |
| [`.vscode/c_cpp_properties.json`](../.vscode/c_cpp_properties.json) · [`.vscode/settings.json`](../.vscode/settings.json) · [`.vscode/extensions.json`](../.vscode/extensions.json) | 配置 Android ARM64 的 C IntelliSense、工作区 Python 环境和扩展推荐。 |

## 用测试和断点读代码

在手机 workspace 和 PJRT bundle 准备好之后，选一个足够小的复跑入口。安装、构建和推送命令见后端 README。 [构建与验证](README.md#build-and-check)

```powershell
$deviceDir = '/data/local/tmp/jax-mali-train.qCUUWZ'
adb shell -T sh "$deviceDir/pjrt/run_on_phone.sh" --abi
adb shell -T sh "$deviceDir/pjrt/run_on_phone.sh" --async
```

- **想看 JAX 调哪一个 C 函数。** 在 VS Code 选 `Mali phone: PJRT plugin in Python`。在 `client_create()` 看握手，在 `loaded_execute()` 看 `argument_lists`、`output_lists` 和 `device_complete_events`。要看实际 socket 发送，在插件的 `execute()` 或 `exchange()` 下断点。 [VS Code 调试说明](README.md#vs-code-调试)
- **想看 GPU 真正拿到什么。** 选 `Mali phone: Android worker`，在 `load_module()` 或 `execute_module()` 下断点。按 F5 让 worker 继续，再运行任务 `Mali phone: run client against debug worker`。`execute_module()` 里看 `iree_runtime_call_invoke()` 前后的输入句柄、IREE 状态和输出 view。要从 `open_vulkan()` 开始单步，使用手动 LLDB 启动流程。 [手动 LLDB](README.md#android-调试入门)
- **想验证队列里的引用。** 从 `abi_smoke.c` 的 `check_async()` 读。它暂停自己的 worker，提交两个有依赖的任务，在任务未完成时删除输入和 executable，再恢复 worker 检查输出。 [C ABI 测试](abi_smoke.c)
- **想看重启后的缓存加载。** 用 `--cache`，从 `check_cache()` 追到 `executable_serialize()`、`read_serialized_image()` 和 `load_image()`。第二个进程移除编译器路径后仍能用保存的产物执行，这是判断确实复用镜像的依据。 [缓存 smoke](smoke.py) · [Python 调用示例](README.md#reuse-a-compiled-executable-after-restart)

第一遍读完，可以试着回答三个问题。`y` 返回时，插件手上有哪些元数据，哪些句柄还没拿到？`z` 在什么地方等 `y`？现在删除 `x`，最后由谁发 `MALI_RELEASE`？三个答案分别落在 `loaded_execute()`、派发线程的 `execute()` 和 `release_resource()`。
