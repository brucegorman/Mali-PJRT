"""Build the Linux arm64 PJRT plugin and Android arm64 worker."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request


SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parent
XLA_REVISION = "47a47140b5f7a33d86901316d9c569de179bae08"  # JAX 0.6.1
PJRT_HEADER_SHA256 = "06d4ed5d94b019b793b9c9723eb60bc26c801b131d77d980bbaa0df63fbf46b5"


def pjrt_header(directory):
    directory.mkdir(parents=True, exist_ok=True)
    header = directory / "pjrt_c_api.h"
    if not header.is_file():
        url = f"https://raw.githubusercontent.com/openxla/xla/{XLA_REVISION}/xla/pjrt/c/pjrt_c_api.h"
        with urllib.request.urlopen(url, timeout=20) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != PJRT_HEADER_SHA256:
            raise RuntimeError("Downloaded PJRT header checksum mismatch")
        header.write_bytes(data)
    elif hashlib.sha256(header.read_bytes()).hexdigest() != PJRT_HEADER_SHA256:
        raise RuntimeError(f"Cached PJRT header checksum mismatch: {header}")
    return header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zig", default=shutil.which("zig"), help="Zig executable on the host")
    parser.add_argument("--debug", action="store_true", help="Build with LLDB symbols")
    parser.add_argument("--output", type=Path, help="Use a separate package directory")
    parser.add_argument("--ndk", type=Path,
                        default=ROOT.parent / ".gpu-android-tools" / "android-ndk-r30")
    parser.add_argument("--iree", type=Path, default=ROOT / "build/iree-3.11-src",
                        help="Prepared IREE v3.11 source, with runtime submodules")
    parser.add_argument("--host-bin", type=Path, default=ROOT / "build/host-bin",
                        help="Directory containing the IREE host generators")
    args = parser.parse_args()
    if not args.zig:
        parser.error("Set --zig or put Zig on PATH")
    toolchain = args.ndk.resolve() / "build/cmake/android.toolchain.cmake"
    if not toolchain.is_file():
        parser.error(f"Android NDK toolchain not found: {toolchain}")
    if not (args.iree / "CMakeLists.txt").is_file() or not args.host_bin.is_dir():
        parser.error("Prepare IREE source and host tools using the root README first")

    header = pjrt_header(ROOT / "build/pjrt-deps")
    output = (args.output.resolve() if args.output is not None else
              ROOT / ("build/pjrt-debug" if args.debug else "build/pjrt"))
    package = output / "jax_plugins/mali"
    package.mkdir(parents=True, exist_ok=True)
    zig_env = {
        **os.environ,
        "ZIG_GLOBAL_CACHE_DIR": str(ROOT / "build/pjrt-deps/zig-global-cache"),
        "ZIG_LOCAL_CACHE_DIR": str(ROOT / "build/pjrt-deps/zig-local-cache"),
    }
    compiler_flags = ["-O0", "-g"] if args.debug else ["-O2"]
    compiler = [
        args.zig, "cc", "-target", "aarch64-linux-gnu.2.28", "-shared", "-fPIC",
        "-std=c11", *compiler_flags, "-pthread", "-fvisibility=hidden", "-Wall", "-Wextra", "-Werror",
        "-I", str(header.parent),
    ]
    subprocess.run([
        *compiler, *(str(SOURCE / file) for file in
                     ("plugin.c", "async.c", "buffer.c", "executable.c", "executable_image.c",
                      "compile_options.c")),
        "-o", str(package / "libpjrt_mali.so"),
    ], check=True, cwd=ROOT, env=zig_env)
    subprocess.run([
        *(flag for flag in compiler if flag not in ("-shared", "-fPIC")),
        str(SOURCE / "abi_smoke.c"), "-ldl", "-o", str(output / "abi_smoke"),
    ], check=True, cwd=ROOT, env=zig_env)

    android_build = ROOT / ("build/pjrt-iree-debug" if args.debug else "build/iree-3.11-android")
    build_type = "Debug" if args.debug else "Release"
    worker_env = {**os.environ, "PATH": str(args.host_bin.resolve()) + os.pathsep + os.environ["PATH"]}
    subprocess.run([
        "cmake", "-S", str(args.iree.resolve()), "-B", str(android_build), "-G", "Ninja",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain.as_posix()}",
        "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-29",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        "-DIREE_BUILD_COMPILER=OFF", "-DIREE_BUILD_TESTS=OFF", "-DIREE_BUILD_SAMPLES=OFF",
        "-DIREE_BUILD_BENCHMARKS=OFF", "-DIREE_HAL_DRIVER_DEFAULTS=OFF",
        "-DIREE_HAL_DRIVER_VULKAN=ON", "-DIREE_ERROR_ON_MISSING_SUBMODULES=OFF",
        f"-DIREE_CMAKE_PLUGIN_PATHS={SOURCE.as_posix()}",
        f"-DMALI_PJRT_HEADER_DIR={header.parent.as_posix()}",
    ], check=True, env=worker_env)
    subprocess.run(["cmake", "--build", str(android_build), "--target", "mali_worker", "-j", "8"],
                   check=True, env=worker_env)
    shutil.copy2(android_build / "tools/mali_worker", output)
    shutil.copy2(SOURCE / "jax_plugins/mali/__init__.py", package)
    for name in ("run_on_phone.sh", "smoke.py"):
        shutil.copy2(SOURCE / name, output)
    shutil.copy2(ROOT / "smoke.py", output / "training_fixture.py")
    print(output)


if __name__ == "__main__":
    main()
