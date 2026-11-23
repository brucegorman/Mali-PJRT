#!/system/bin/sh
set -eu

cd "$(dirname "$0")"
package="$PWD"
workspace="$(dirname "$package")"
glibc="$workspace/glibc/usr/lib/aarch64-linux-gnu"
python_home="$workspace/python"
export PYTHONHOME="$python_home"
export PYTHONPATH="$package:$workspace/site-packages"
export TMPDIR="$workspace/tmp"
export JAX_PLATFORMS=mali
export MALI_GLIBC_LOADER="$glibc/ld-linux-aarch64.so.1"
export MALI_GLIBC_LIBRARIES="$glibc:$python_home/lib"
export MALI_IREE_COMPILER="$workspace/site-packages/iree/compiler/_mlir_libs/iree-compile"

chmod 700 "$package/mali_worker"
exec "$glibc/ld-linux-aarch64.so.1" --library-path "$glibc:$python_home/lib" \
    "$python_home/bin/python3.11" "$package/smoke.py" "$@"
