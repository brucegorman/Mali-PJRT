# Mali PJRT scaffold

JAX 0.6.1 loads a Linux arm64 PJRT library and connects to an Android arm64
worker over an abstract Unix socket. The worker opens a Vulkan logical device
and compute queue. JAX exposes that device through `jax.devices("mali")`.

| File | Entry point |
|---|---|
| [Python registration](jax_plugins/mali/__init__.py) | JAX discovers the package and loads `libpjrt_mali.so`. |
| [PJRT C plugin](plugin.c) | `GetPjrtApi()` exposes the API table. `client_create()` reads and owns the device description. |
| [Android worker](worker.c) | `open_vulkan()` opens an Arm Vulkan 1.1 device. `serve_client()` holds it until the PJRT client disconnects. |
| [Protocol](protocol.h) | The versioned hello, device metadata packet, and abstract socket address helper. |

The Y36m reports a single `Mali-G57 MC2` with Vulkan 1.1.177. The default PJRT
device memory is described to JAX. Buffer transfer, JIT compilation, and GPU
execution are the remaining backend operations.

## Build and check

From the project directory, with Python, CMake, Ninja, and Zig available:

```powershell
python pjrt/build.py --zig build/pjrt-deps/zig-x86_64-windows-0.15.2/zig.exe
$deviceDir = '/data/local/tmp/jax-mali-train.qCUUWZ'
adb push build/pjrt "$deviceDir/"
adb shell -T sh "$deviceDir/pjrt/run_on_phone.sh"
adb shell -T sh "$deviceDir/pjrt/run_on_phone.sh" --missing-worker
```

Pass `--ndk` to override the shared Android NDK location. If Zig is on PATH,
omit `--zig`. Use the device directory set up in the
[phone training instructions](../README.md#compile-and-train-on-the-phone).
The build fetches and verifies the PJRT header from the XLA revision used by
JAX 0.6.1. The plugin targets glibc 2.28, and the worker targets Android API 24.

The smoke check verifies one local Mali device with real Vulkan metadata,
the default memory description, and clean worker shutdown after the JAX
process exits. It checks the PJRT errors for operations still in development
and for a missing worker.

## Android 调试入门

ADB 负责连手机、传文件和转发端口。Windows 上的 LLDB 读取本地带符号的二进制，手机上的 arm64 `lldb-server` 负责暂停进程和读取内存。两端的二进制要来自同一次构建。

- 查 Vulkan 设备、队列或 worker 的 socket，用下面的 worker 调试流程。
- 查 JAX 调进 C 回调的过程，用 [Python 进程里的插件调试流程](#调试-python-进程里的-pjrt-插件)。

### 在 Windows 终端一启动手机上的 worker

先确认 `adb devices -l` 把手机列为 `device`。如果显示 `unauthorized`，在手机上接受 USB 调试授权。`--debug` 会生成带符号的 `build/pjrt-debug` 包。手机上已经有同一次构建的调试包和 `lldb-server` 时，可以从 `adb forward` 开始。

```powershell
cd D:\files\jax-mali
adb devices -l

python pjrt/build.py --debug --zig build/pjrt-deps/zig-x86_64-windows-0.15.2/zig.exe
$deviceDir = '/data/local/tmp/jax-mali-train.qCUUWZ'
$llvm = 'D:/files/.gpu-android-tools/android-ndk-r30/toolchains/llvm/prebuilt/windows-x86_64'
adb push build/pjrt-debug "$deviceDir/"
adb push "$llvm/lib/clang/21/lib/linux/aarch64/lldb-server" "$deviceDir/"
adb shell chmod 700 "$deviceDir/lldb-server" "$deviceDir/pjrt-debug/mali_worker"
adb forward --no-rebind tcp:49183 tcp:49183
adb shell -T "$deviceDir/lldb-server" gdbserver 127.0.0.1:49183 -- "$deviceDir/pjrt-debug/mali_worker" mali-debug-lldb
```

最后一行会占住终端，因为手机上的服务在等 LLDB。保持窗口打开。

### 在 Windows 终端二打开 LLDB

```powershell
cd D:\files\jax-mali
$llvm = 'D:/files/.gpu-android-tools/android-ndk-r30/toolchains/llvm/prebuilt/windows-x86_64'
& "$llvm/bin/lldb.exe" build/pjrt-debug/mali_worker
```

看到 `(lldb)` 后逐行输入以下命令。

```text
breakpoint set --name open_vulkan
gdb-remote 127.0.0.1:49183
continue
thread backtrace
frame variable vulkan
next
```

这台 Y36m 上实测会停在 [`open_vulkan()`](worker.c)。用 `next` 走到 `vkCreateInstance()` 返回之后，再敲 `frame variable result` 看 Vulkan 返回码。

| 你想做什么 | LLDB 命令 |
|---|---|
| 执行当前行，跳过函数调用 | `next` |
| 单步进入函数 | `step` |
| 跑到当前函数返回 | `finish` |
| 看当前栈帧的局部变量 | `frame variable` |
| 看调用栈 | `bt` |
| 跑到下一个断点 | `continue` |

调完后，在 LLDB 输入 `process kill` 和 `quit`。回到 PowerShell 后运行 `adb forward --remove tcp:49183`，撤掉这次的端口转发。

## 调试 Python 进程里的 PJRT 插件

JAX 会在手机上的 glibc Python 进程里动态加载 `libpjrt_mali.so`。所以先用手机的 glibc loader 作为调试目标，再设置一个等插件加载后才生效的断点。

在 PowerShell 里沿用上面的 `$deviceDir` 和 `$llvm`。

```powershell
adb pull "$deviceDir/glibc/usr/lib/aarch64-linux-gnu/ld-linux-aarch64.so.1" build/ld-linux-aarch64.so.1
adb forward --no-rebind tcp:49184 tcp:49184
adb shell
```

进入 `adb shell` 后，以下命令在手机上执行。

```sh
root=/data/local/tmp/jax-mali-train.qCUUWZ
p="$root/pjrt-debug"
glibc="$root/glibc/usr/lib/aarch64-linux-gnu"
export MALI_PJRT_SOCKET="mali-lldb-$$" JAX_PLATFORMS=mali
export PYTHONHOME="$root/python" PYTHONPATH="$p:$root/site-packages"
"$p/mali_worker" "$MALI_PJRT_SOCKET" &
```

看到 worker 输出 `READY` 后，在手机上启动受 LLDB 控制的 Python。

```sh
"$root/lldb-server" gdbserver 127.0.0.1:49184 -- \
    "$glibc/ld-linux-aarch64.so.1" --library-path "$glibc:$PYTHONHOME/lib" \
    "$PYTHONHOME/bin/python3.11" -c 'import jax; print(jax.devices("mali"))'
```

在另一个 PowerShell 窗口运行 `& "$llvm/bin/lldb.exe" build/ld-linux-aarch64.so.1`，然后在 LLDB 提示符里输入：

```text
settings set target.exec-search-paths build/pjrt-debug/jax_plugins/mali
breakpoint set --name client_create --shlib libpjrt_mali.so
gdb-remote 127.0.0.1:49184
```

如果 LLDB 先停在启动信号，输入 `continue`；如果已经显示 `stop reason = breakpoint`，直接看现场。在 [`client_create()`](plugin.c) 断住时，用 `thread backtrace` 看调用栈，用 `frame variable args` 看 PJRT 参数。继续执行后，Python 应该打印 `[Mali-G57 MC2]`。

想跟踪数组上传，就把断点改成 `client_buffer_from_host`，把 Python 的触发动作改为 `jax.device_put()`。当前上传回调会返回 `UNIMPLEMENTED`。结束插件调试后，在 PowerShell 运行 `adb forward --remove tcp:49184`。
