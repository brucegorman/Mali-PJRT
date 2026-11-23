#!/system/bin/sh
set -eu

cd "$(dirname "$0")"
workspace="$PWD"
glibc="$workspace/glibc/usr/lib/aarch64-linux-gnu"
python_home="$workspace/python"
if [ ! -x "$glibc/ld-linux-aarch64.so.1" ]; then
    mkdir -p glibc
    tar -xzf ubuntu-base.tar.gz -C glibc usr/lib/aarch64-linux-gnu
fi
if [ ! -x "$python_home/bin/python3.11" ]; then
    tar -xzf python.tar.gz
fi
mkdir -p "$workspace/tmp"
chmod 700 "$workspace/iree-run-module"
export PYTHONHOME="$python_home"
export PYTHONPATH="$workspace/site-packages"
export TMPDIR="$workspace/tmp"
export JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

run_linux() {
    "$glibc/ld-linux-aarch64.so.1" --library-path "$glibc:$python_home/lib" "$@"
}

run_python() {
    run_linux "$python_home/bin/python3.11" "$@"
}

if [ ! -f "$workspace/site-packages/.installed" ]; then
    run_python -m pip --disable-pip-version-check install --no-index \
        --no-cache-dir --no-compile --no-deps --target "$workspace/site-packages" \
        "$workspace"/*.whl
    touch "$workspace/site-packages/.installed"
fi

output="$workspace/on-device"
mkdir -p "$output"
printf 'Compiling on Android %s (%s)\n' "$(getprop ro.product.model)" "$(uname -m)"
run_python "$workspace/smoke.py" --export "$output"
compiler="$PYTHONPATH/iree/compiler/_mlir_libs/iree-compile"
run_linux "$compiler" --version
run_linux "$compiler" "$output/train_eight_steps.mlir" \
    --iree-input-type=stablehlo --iree-hal-target-device=vulkan \
    -o "$output/train_eight_steps.vmfb"

"$workspace/iree-run-module" --list_devices=vulkan
"$workspace/iree-run-module" --device=vulkan \
    --module="$output/train_eight_steps.vmfb" --function=main \
    '--input=2xf32=1 2' '--input=2x2xf32=1 0 0 1' '--input=2xf32=3 7' \
    "--output=@$output/weights.npy"

run_python - "$output/weights.npy" <<'PY'
import sys
import numpy as np

weights = np.load(sys.argv[1])
target = np.array([3, 7])
initial = np.array([1, 2])
expected = target + (initial - target) * (7 / 8) ** 8
np.testing.assert_allclose(weights, expected, rtol=1e-6, atol=1e-6)
loss_before = np.mean((initial - target) ** 2)
loss_after = np.mean((weights - target) ** 2)
print(f"PASS: 8 Mali SGD updates, MSE {loss_before:.6f} -> {loss_after:.6f}")
print(f"weights={weights.tolist()}")
PY
