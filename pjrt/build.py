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
    parser.add_argument("--ndk", type=Path,
                        default=ROOT.parent / ".gpu-android-tools" / "android-ndk-r30")
    args = parser.parse_args()
    if not args.zig:
        parser.error("Set --zig or put Zig on PATH")
    toolchain = args.ndk.resolve() / "build/cmake/android.toolchain.cmake"
    if not toolchain.is_file():
        parser.error(f"Android NDK toolchain not found: {toolchain}")

    header = pjrt_header(ROOT / "build/pjrt-deps")
    output = ROOT / ("build/pjrt-debug" if args.debug else "build/pjrt")
    package = output / "jax_plugins/mali"
    package.mkdir(parents=True, exist_ok=True)
    zig_env = {
        **os.environ,
        "ZIG_GLOBAL_CACHE_DIR": str(ROOT / "build/pjrt-deps/zig-global-cache"),
        "ZIG_LOCAL_CACHE_DIR": str(ROOT / "build/pjrt-deps/zig-local-cache"),
    }
    compiler_flags = ["-O0", "-g"] if args.debug else ["-O2"]
    subprocess.run([
        args.zig, "cc", "-target", "aarch64-linux-gnu.2.28", "-shared", "-fPIC",
        "-std=c11", *compiler_flags, "-fvisibility=hidden", "-Wall", "-Wextra", "-Werror",
        "-I", str(header.parent), str(SOURCE / "plugin.c"),
        "-o", str(package / "libpjrt_mali.so"),
    ], check=True, cwd=ROOT, env=zig_env)

    android_build = ROOT / ("build/pjrt-android-debug" if args.debug else "build/pjrt-android")
    build_type = "Debug" if args.debug else "Release"
    subprocess.run([
        "cmake", "-S", str(ROOT), "-B", str(android_build), "-G", "Ninja",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain.as_posix()}",
        "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-24",
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ], check=True)
    subprocess.run(["cmake", "--build", str(android_build), "--target", "mali_worker"],
                   check=True)
    shutil.copy2(android_build / "mali_worker", output)
    shutil.copy2(SOURCE / "jax_plugins/mali/__init__.py", package)
    for name in ("run_on_phone.sh", "smoke.py"):
        shutil.copy2(SOURCE / name, output)
    print(output)


if __name__ == "__main__":
    main()
