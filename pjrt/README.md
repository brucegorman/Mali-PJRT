# Mali PJRT backend

[中文源码阅读指南](READING_GUIDE.md) · [错误与复现记录](BUGS.md)

[Valhall 矩阵乘与激活存储](../benchmarks/README.md#valhall-矩阵乘和激活存储)

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
| [PJRT buffer callbacks](buffer.c) | Transfers bytes in chunks and exposes buffer metadata and readiness. |
| [PJRT executable callbacks](executable.c) | Invokes the arm64 IREE compiler, loads an executable, and builds queued requests with pending outputs. |
| [Asynchronous execution](async.c) | Owns the single-device FIFO, completion events, callback thread, and references to in-flight worker resources. |
| [Executable images](executable_image.c) | Serializes VMFB with a versioned header and validates device, driver, compiler revision, and checksum on reload. |
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

IREE manages Vulkan allocation and transfer. `Execute` returns output buffer
metadata and a pending event after enqueueing. A dispatch thread runs requests
in FIFO order, so a pending output can be used by the next execution on the
same device. The job holds references to its inputs, outputs, and executable
until completion, including after their public handles are deleted. `Await`
blocks on the event, and pending `OnReady` callbacks run on a separate thread.
Worker failures complete the corresponding event with an error and propagate
to its dependents. Host transfers still run synchronously, and a readback
waits for the producing event before taking the socket lock. The Android
worker runs one IREE invocation at a time. Host send/receive callbacks and
distributed execution are outside this implementation.

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
adb shell -T sh "$deviceDir/pjrt/run_on_phone.sh" --async
adb shell -T sh "$deviceDir/pjrt/run_on_phone.sh" --cache
adb shell -T sh "$deviceDir/pjrt/run_on_phone.sh" --missing-worker
adb shell -T sh "$deviceDir/pjrt/run_on_phone.sh" --disconnect
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
exercises backend initialization failure. The `--async` C ABI check pauses its
worker and submits two dependent executions. It checks pending events and
output metadata, deletes the original inputs and executable, then resumes
the worker and checks the readback. A completion callback also waits on the
second event. The `--disconnect` JAX check queues two dependent calls while
its worker is paused, then kills the worker and checks that both waits and a
readback report `UNAVAILABLE` with operation context.

### Probe Vulkan fence submission

On Mali-G57, run `adb shell -T sh "$deviceDir/pjrt/run_on_phone.sh" --fences`.
The probe compiles two StableHLO exports on the phone with
`--iree-execution-model=async-external`. The Android worker calls each export
with an IREE wait/signal fence, passes the intermediate device buffer directly
to the second export, waits on the final fence, and checks `[5, 7, 9, 11]`.

The start fence is initially unsignaled. The probe releases it when both
calls have returned and both completion fences are still pending. A one-second
watchdog releases it if submission blocks. Exit `0` confirms pending returns,
`3` means the watchdog released the gate and the numerical result was correct,
and other nonzero exits indicate a probe failure. The pinned Vulkan runtime
on the Y36m returns `3`. `submit_first_ms` includes any watchdog delay, and
`final_wait_ms` measures only the explicit wait after opening the gate.

## Reuse a compiled executable after restart

Compile once in Python on the phone and save the executable bytes:

```python
from pathlib import Path

import jax
from training_fixture import train_eight_steps

compiled = jax.jit(train_eight_steps).lower(weights, inputs, targets).compile()
device = weights.device
Path("train.malivmfb").write_bytes(
    device.client.serialize_executable(compiled.runtime_executable())
)
```

In a new JAX process with the worker running, upload the inputs, then load and
execute the saved image:

```python
from pathlib import Path

import jax

device, = jax.devices("mali")
executable = device.client.deserialize_executable(
    Path("train.malivmfb").read_bytes(), executable_devices=[device]
)
weights, = executable.execute([weights, inputs, targets])
```

The arguments in the second example are arrays placed on that process's
Mali device. The loader checks the executable format version, IREE revision,
Vulkan driver and pipeline cache UUID, and a checksum of the VMFB bytes.
The pinned JAX 0.6.1 automatic cache has a platform allowlist that excludes
`mali`. Use the explicit save and load calls for this version.

The `--cache` smoke test writes an image, exits, and starts a fresh worker and
Python process. The second process unsets the compiler path, rejects altered
images, and runs the saved training twice without compiling.

## VS Code 调试

在 VS Code 打开 `jax-mali` 根目录。工作区配置使用 `.venv/Scripts/python.exe`，并为 C IntelliSense 提供 Android NDK、PJRT 和 IREE 的头文件路径。如果 VS Code 已为此工作区记住其他解释器，运行 `Python: Select Interpreter` 选择项目的 `.venv`。

在 **Run and Debug** 里按需选择一个配置。

| 配置 | 怎么用 |
|---|---|
| `Python: CPU training smoke` | 在电脑上用 `.venv` 运行根目录的 `smoke.py`。可在 `train_step()` 打 Python 断点。 |
| `Mali phone: PJRT plugin in Python` | 启动手机上的 worker，并用 CodeLLDB 调试手机 Python 加载的 C 插件。可在 `client_create()` 或 `loaded_execute()` 打断点。从入口暂停处按 F5 继续。 |
| `Mali phone: Android worker` | 用 CodeLLDB 附加到已打开 Vulkan 的 worker。可在 `serve_client()` 后面的请求处理代码或 `execute_module()` 打断点。按 F5 继续，再用 **Tasks: Run Task** 运行 `Mali phone: run client against debug worker`。需要调试更早的 `open_vulkan()` 时，用下节的手动启动方式。 |

手机配置自动构建并推送带符号的 `build/pjrt-vscode`、启动 Android `lldb-server`，并通过 ADB 转发本机端口 `49185`。一次运行一个手机调试会话。两种手机配置都使用已准备好的手机 Python 和编译器环境。如果手机工作目录不同，在启动 VS Code 的环境中设置 `JAX_MALI_PHONE_DIR`。手动命令见下节。

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
