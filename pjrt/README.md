# Mali PJRT backend

[错误与复现记录](BUGS.md)

JAX 0.6.1 loads a Linux arm64 PJRT library and connects to an Android arm64
worker over an abstract Unix socket. The worker opens a Vulkan logical device
and compute queue. JAX exposes that device through `jax.devices("mali")`.
Use `jax.device_put(x, mali)` to upload and `np.asarray(x_on_mali)` to read back.
Calling a JAX JIT function compiles StableHLO with IREE on the phone. The
worker loads the resulting VMFB and runs it on Vulkan. It returns buffer
handles so a caller can pass the results into another JIT call.

| File | Entry point |
|---|---|
| [Python registration](jax_plugins/mali/__init__.py) | JAX discovers the package and loads `libpjrt_mali.so`. |
| [PJRT C plugin](plugin.c) | `GetPjrtApi()` exposes the API table. `client_create()` reads and owns the device description. |
| [Android worker](worker.c) | `open_vulkan()` opens an Arm Vulkan 1.1 device. `serve_client()` holds it until the PJRT client disconnects. |
| [PJRT buffer callbacks](buffer.c) | Transfers bytes in chunks and keeps the worker handle, shape, and dtype. |
| [PJRT executable callbacks](executable.c) | Invokes the arm64 IREE compiler, loads an executable, and dispatches it by handle. |
| [Compile options](compile_options.c) | Validates the pinned XLA options for one local device before starting IREE. |
| [Output metadata](output_metadata.c) | Reads static output shapes and dtypes from IREE reflection, including unsigned types recorded in the ABI encoding. |
| [Worker buffers and execution](worker_buffers.c) | Holds the Vulkan HAL buffer views and the IREE sessions that run compiled modules. |
| [Protocol](protocol.h) | Device metadata, tensor descriptions, transfers, and executable requests. |

The Y36m reports a single `Mali-G57 MC2` with Vulkan 1.1.177. The backend
supports one device, dense row-major buffers, strided host uploads, and
conventional byte-aligned buffer types such as float32, int32, and bool.
On Mali-G57 it explicitly compiles for IREE's `mali-g57` target. The generic
Vulkan target produced incorrect FP16 values on this phone in the direct
IREE runner. The device-specific target passed the FP16 JIT checks.

The backend accepts the pinned JAX defaults and a single device assignment to
device 0. Custom XLA compiler overrides, explicit XLA layouts, multi-device
options and unsupported output signatures return errors. Output dtypes and
shapes are available from PJRT before execution and checked against the
runtime buffers after a call.

IREE manages Vulkan allocation and transfer, and the PJRT callbacks return
completed events for synchronous operations. Executable serialization,
host callbacks, and distributed execution are outside this implementation.

## Build and check

With Python, CMake, Ninja, and Zig available, prepare the IREE v3.11 source
and host generators using the [root build instructions](../README.md#build-the-android-runtime).
Then run from the project directory:

```powershell
python pjrt/build.py --zig build/pjrt-deps/zig-x86_64-windows-0.15.2/zig.exe
$deviceDir = '/data/local/tmp/jax-mali-train.qCUUWZ'
adb push build/pjrt "$deviceDir/"
adb shell -T sh "$deviceDir/pjrt/run_on_phone.sh"
adb shell -T sh "$deviceDir/pjrt/run_on_phone.sh" --abi
adb shell -T sh "$deviceDir/pjrt/run_on_phone.sh" --missing-worker
```

Pass `--ndk`, `--iree`, or `--host-bin` to override the existing Android
NDK, IREE source, or host generator location. If Zig is on PATH, omit `--zig`.
Use the device directory set up in the
[phone training instructions](../README.md#compile-and-train-on-the-phone).
The build fetches and verifies the PJRT header from the XLA revision used by
JAX 0.6.1. The plugin targets glibc 2.28, and the IREE worker targets Android API 29.
If LLDB has the existing worker open on Windows, use `--debug --output
build/pjrt-jit-debug` and use `pjrt-jit-debug` as the package directory
in the debug commands below.

The smoke check verifies device metadata, upload, readback and deletion for
a multi-packet float32 matrix, a strided int32 view, a scalar, an empty array,
and bool values. It mutates the host input after upload to verify independent
storage. It verifies JIT addition, multiple outputs, batched matrix
multiplication, transpose/slice/reduction, unsigned integers, booleans,
FP16, scalars and empty arrays. Two eight-step training calls keep the
intermediate weights on Mali and reduce the loss from 14.500000 to 0.202128.
The C ABI test queries output shapes and types before execution and checks
rejection of malformed options, extra replicas or partitions, the wrong
device assignment, and custom XLA debug options. The missing-worker check
exercises backend initialization failure.

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

想跟踪数组上传，就在 [buffer.c](buffer.c) 的 `client_buffer_from_host` 下断点，用 `jax.device_put()` 触发。跟踪读回时，在 `buffer_to_host` 下断点，调用 `np.asarray()`。结束插件调试后，在 PowerShell 运行 `adb forward --remove tcp:49184`。
