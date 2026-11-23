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
