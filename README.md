# JAX on Mali

The Android training test runs JAX tracing and IREE compilation on the phone's arm64 CPU, then executes the resulting graph on Mali. Python and the IREE compiler run through a local arm64 glibc loader. The Android IREE runner uses the phone's native Vulkan driver. Direct JAX JIT through a single PJRT process remains a separate milestone.

## Compile and train on the phone

On the vivo Y36m, [train_on_phone.sh](train_on_phone.sh) exported StableHLO from [smoke.py](smoke.py), compiled it to Vulkan with IREE v3.11.0, and ran eight SGD updates on Mali. Mean squared error went from **14.500000** to **1.711973**. The script checked the final weights against the analytical solution and saved them to `on-device/weights.npy` inside the device workspace.

The prepared device workspace can be reused with:

```powershell
adb shell -T sh /data/local/tmp/jax-mali-train.qCUUWZ/train_on_phone.sh
```

To stage a fresh workspace, use the same Android runner built below and the arm64 inputs cached under `build/phone-assets`. These are Ubuntu Base 24.04.5 for glibc, Astral's standalone CPython 3.11.18 for Linux aarch64, and Linux aarch64 or pure Python wheels for JAX 0.6.1 and IREE v3.11.0 and their dependencies. The script installs the wheels offline and generates the StableHLO and VMFB files on the phone on every run.

```powershell
$deviceDir = (adb shell mktemp -d /data/local/tmp/jax-mali-train.XXXXXX).Trim()
$assets = (Get-ChildItem build/phone-assets -File).FullName
adb push @assets "$deviceDir/"
adb push smoke.py train_on_phone.sh build/iree-3.11-android/tools/iree-run-module "$deviceDir/"
adb shell -T sh "$deviceDir/train_on_phone.sh"
```

[Ubuntu Base archives](https://cdimage.ubuntu.com/ubuntu-base/releases/24.04/release/) · [Astral CPython builds](https://github.com/astral-sh/python-build-standalone/releases/tag/20261110)

## First device

The vivo Y36m (V2318A, Android API 33) reports Mali-G57 MC2 with a compute queue. Its Vulkan loader reports **1.3.0**, and its physical device reports **1.1.177**. IREE v3.14.0 requires Vulkan 1.3 and specific features on the physical device. IREE v3.11.0 works for these fixtures because its Android Vulkan driver requests Vulkan 1.1.

The v3.11.0 Android runner passed both checks on the Y36m with `--device=vulkan`. The forward result matched `[5, 11]`, and the gradient update matched `[-0.75, -0.5]`. JAX 0.6.1 generated both StableHLO graphs on the Windows host. The runner was built from IREE commit `e4a3b0405d7d23554da26403658d0e8c3c5ecf25` with Vulkan as its only registered HAL driver.

[IREE Vulkan requirements](https://iree.dev/guides/deployment-configurations/gpu-vulkan/) · [IREE v3.11.0 Android driver](https://github.com/iree-org/iree/blob/v3.11.0/runtime/src/iree/hal/drivers/vulkan/registration/driver_module.cc)

## Device probe

From PowerShell with the workspace's Android NDK r30, CMake, Ninja and ADB installed:

```powershell
$ndk = (Resolve-Path ../.gpu-android-tools/android-ndk-r30).Path
cmake -S . -B build/android -G Ninja "-DCMAKE_TOOLCHAIN_FILE=$ndk/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24
cmake --build build/android
adb push build/android/vulkan_probe /data/local/tmp/jax_mali_probe
adb shell chmod 700 /data/local/tmp/jax_mali_probe
adb shell /data/local/tmp/jax_mali_probe
adb shell rm -f /data/local/tmp/jax_mali_probe
```

The probe exits with 0 when a Mali device meets the Vulkan 1.3 baseline, 2 when enumeration succeeds and no Mali device meets the baseline, and 1 on a loader or enumeration error. The probe's baseline check is a prerequisite check. Actual execution is verified separately.

## Compile the diagnostic fixtures on a host

Use Python 3.11 and match the IREE compiler release to the Android runtime release. The fixtures are a matrix-vector product with expected output `[5, 11]` and a mean-squared-error SGD step with expected updated weights `[-0.75, -0.5]`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install 'jax[cpu]==0.6.1' 'iree-base-compiler==3.11.0'
.\.venv\Scripts\python.exe smoke.py --platform cpu
.\.venv\Scripts\python.exe smoke.py --export build/stablehlo
foreach ($name in @('predict', 'train_step')) {
    .\.venv\Scripts\iree-compile.exe "build/stablehlo/$name.mlir" --iree-input-type=stablehlo --iree-hal-target-device=vulkan -o "build/$name-v3.11.vmfb"
    if ($LASTEXITCODE -ne 0) { throw "Failed to compile $name" }
}
```

## Build the Android runtime

The workspace checkout at `../iree-src` has the v3.11.0 tag. The following prepares a release-matched worktree and builds the Vulkan-only runner. MinGW builds the schema and data generators for Windows, and the Android NDK builds the device executable.

```powershell
$ndk = (Resolve-Path ../.gpu-android-tools/android-ndk-r30).Path
if (!(Test-Path build/iree-3.11-src/.git)) {
    git -C ../iree-src worktree add --detach ../jax-mali/build/iree-3.11-src v3.11.0
}
git -C build/iree-3.11-src submodule update --init --depth 1 third_party/flatcc third_party/vulkan_headers third_party/benchmark third_party/printf
cmake -S build/iree-3.11-src/third_party/flatcc -B build/flatcc-3.11-host -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release -DFLATCC_TEST=OFF '-DCMAKE_POLICY_VERSION_MINIMUM=3.5'
cmake --build build/flatcc-3.11-host --target flatcc_cli
New-Item -ItemType Directory -Force build/host-bin | Out-Null
Copy-Item build/iree-3.11-src/third_party/flatcc/bin/flatcc.exe build/host-bin/iree-flatcc-cli.exe
g++ -O2 -std=c++17 build/iree-3.11-src/build_tools/embed_data/iree-c-embed-data-main.cc -o build/host-bin/iree-c-embed-data.exe
$env:PATH = "$(Resolve-Path build/host-bin);$env:PATH"
cmake -S build/iree-3.11-src -B build/iree-3.11-android -G Ninja `
    "-DCMAKE_TOOLCHAIN_FILE=$ndk/build/cmake/android.toolchain.cmake" `
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 -DCMAKE_BUILD_TYPE=Release `
    -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_BENCHMARKS=OFF `
    -DIREE_BUILD_SAMPLES=OFF -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_VULKAN=ON `
    -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF
cmake --build build/iree-3.11-android --target iree-run-module
python verify_android.py --runner build/iree-3.11-android/tools/iree-run-module `
    --predict build/predict-v3.11.vmfb --train-step build/train_step-v3.11.vmfb
```

## PJRT milestone

The [PJRT backend](pjrt/README.md) connects the JAX plugin loader to an
Android worker. It reuses the phone's glibc JAX environment and uses a Unix
socket to reach the Android process. On the Y36m, `jax.devices("mali")` returns
the Mali-G57 MC2 opened by the worker.

The JIT milestone is to run the `train_eight_steps` fixture through the
`mali` backend. IREE's experimental
`iree_vulkan` plugin provides examples of PJRT compilation, device buffers,
and execution.

[IREE PJRT build guide](https://github.com/iree-org/iree/tree/v3.11.0/integrations/pjrt) · [JAX platform matrix](https://docs.jax.dev/en/latest/installation.html)
