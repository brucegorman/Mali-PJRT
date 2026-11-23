"""Verify Mali PJRT device discovery, buffers, and JIT training on Android."""

import argparse
import os
from pathlib import Path
import select
import subprocess
import uuid

def check_device(missing_worker):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from training_fixture import train_eight_steps

    jax.config.update("jax_platforms", "mali")
    if missing_worker:
        try:
            jax.devices("mali")
        except RuntimeError as error:
            if "Could not handshake with Mali worker" not in str(error):
                raise
            print("PASS: missing worker returned an availability error", flush=True)
            return
        raise RuntimeError("Mali was available without its worker")

    devices = jax.devices("mali")
    if len(devices) != 1 or jax.local_devices(backend="mali") != devices:
        raise RuntimeError(f"Expected one local Mali device, got {devices}")
    device = devices[0]
    if (device.platform, device.id, device.process_index, device.vendor_id) != (
        "mali", 0, 0, 0x13B5,
    ):
        raise RuntimeError(f"Unexpected device metadata: {device!r}")
    if not device.device_kind or device.default_memory().kind != "device":
        raise RuntimeError(f"Missing device kind or default memory: {device!r}")
    print(f"PASS: jax.devices('mali') -> {devices}", flush=True)
    print(f"vendor=0x{device.vendor_id:x} device=0x{device.device_id:x} "
          f"compute_queue_family={device.queue_family} {device.client.platform_version}",
          flush=True)

    cases = (
        np.arange(257 * 257, dtype=np.float32).reshape(257, 257),
        np.arange(35, dtype=np.int32).reshape(5, 7)[::-1, ::2].T,
        np.array(3.5, dtype=np.float32),
        np.empty((0, 3), dtype=np.float16),
        np.array([True, False, True]),
    )
    for host in cases:
        expected = host.copy()
        array = jax.device_put(host, device)
        array.block_until_ready()
        if array.device != device or array.on_device_size_in_bytes() != host.nbytes:
            raise RuntimeError("Unexpected Mali buffer device or byte length")
        host[...] = 0
        array.copy_to_host_async()
        np.testing.assert_array_equal(np.asarray(array), expected, strict=True)
        array.delete()
        array.delete()
        if not array.is_deleted():
            raise RuntimeError("Mali buffer was not marked deleted")
    print("PASS: Mali upload, readback, and deletion for five input layouts", flush=True)

    x = jax.device_put(np.array([1, 2], dtype=np.float32), device)
    doubled = jax.jit(lambda values: values + values)(x)
    if doubled.device != device:
        raise RuntimeError("JIT addition did not produce a Mali array")
    np.testing.assert_array_equal(np.asarray(doubled), np.array([2, 4], dtype=np.float32))
    print("PASS: JAX JIT addition on Mali", flush=True)
    first, second = jax.jit(lambda values: (values + 1, values * 3))(x)
    if first.device != device or second.device != device:
        raise RuntimeError("JIT returned a result on the wrong device")
    np.testing.assert_array_equal(np.asarray(first), np.array([2, 3], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(second), np.array([3, 6], dtype=np.float32))
    print("PASS: JAX JIT returned two independent Mali outputs", flush=True)

    matrix = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    matrix_operand = np.arange(8, dtype=np.float32).reshape(4, 2)
    matrix_result = jax.jit(jnp.matmul)(
        jax.device_put(matrix, device), jax.device_put(matrix_operand, device)
    )
    np.testing.assert_array_equal(np.asarray(matrix_result), matrix @ matrix_operand, strict=True)

    data = np.arange(24, dtype=np.int32).reshape(2, 3, 4) - 10
    reduce = jax.jit(lambda value: jnp.sum(value.transpose(2, 0, 1)[1:4], axis=-1))
    np.testing.assert_array_equal(
        np.asarray(reduce(jax.device_put(data, device))),
        data.transpose(2, 0, 1)[1:4].sum(axis=-1, dtype=np.int32),
        strict=True,
    )
    for values, operation, expected in (
        (np.array([0, 1, 0x80000000], dtype=np.uint32), jnp.bitwise_not,
         np.array([0xffffffff, 0xfffffffe, 0x7fffffff], dtype=np.uint32)),
        (np.array([True, False, True]), jnp.logical_not, np.array([False, True, False])),
        (np.array([[0.5, 1.25]], dtype=np.float16), lambda value: value + value,
         np.array([[1, 2.5]], dtype=np.float16)),
        (np.array([-0.5, 1.25, 2.5], dtype=np.float16), lambda value: value + value,
         np.array([-1, 2.5, 5], dtype=np.float16)),
        (np.array(2.0, dtype=np.float32), lambda value: value * value,
         np.array(4, dtype=np.float32)),
        (np.empty((0, 3), dtype=np.float32), lambda value: value + value,
         np.empty((0, 3), dtype=np.float32)),
    ):
        result = jax.jit(operation)(jax.device_put(values, device))
        if result.device != device:
            raise RuntimeError("JIT compatibility check returned an array on the wrong device")
        np.testing.assert_array_equal(np.asarray(result), expected, strict=True)
    print("PASS: batched matmul, transpose/slice/reduce, uint32, bool, float16, scalar, and empty JIT",
          flush=True)

    try:
        jax.jit(lambda value: value + 3,
                compiler_options={"xla_backend_optimization_level": 1})(x)
    except RuntimeError as error:
        if "Mali does not support XLA compiler option overrides" not in str(error):
            raise
    else:
        raise RuntimeError("Mali accepted an unsupported compiler override")
    print("PASS: unsupported JAX compiler override rejected", flush=True)

    initial = np.array([1, 2], dtype=np.float32)
    target = np.array([3, 7], dtype=np.float32)
    weights = jax.device_put(initial, device)
    inputs = jax.device_put(np.eye(2, dtype=np.float32), device)
    targets = jax.device_put(target, device)
    train = jax.jit(train_eight_steps)
    for _ in range(2):
        weights = train(weights, inputs, targets)
        if weights.device != device:
            raise RuntimeError("JIT training did not return Mali weights")
    final = np.asarray(weights)
    expected = target + (initial - target) * (7 / 8) ** 16
    np.testing.assert_allclose(final, expected, rtol=1e-6, atol=1e-6)
    mse_before = np.mean((initial - target) ** 2)
    mse_after = np.mean((final - target) ** 2)
    print(f"PASS: two Mali JIT training calls, MSE {mse_before:.6f} -> {mse_after:.6f}",
          flush=True)
    print(f"weights={final.tolist()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--missing-worker", action="store_true")
    mode.add_argument("--abi", action="store_true", help="Check compile options and output metadata via C")
    parser.add_argument("--client", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.client:
        check_device(args.missing_worker)
        return

    package = Path(__file__).resolve().parent
    socket_name = f"mali-pjrt-{uuid.uuid4().hex}"
    os.environ["MALI_PJRT_SOCKET"] = socket_name
    if args.missing_worker:
        check_device(missing_worker=True)
        return

    worker = subprocess.Popen([str(package / "mali_worker"), socket_name],
                              stdout=subprocess.PIPE, text=True)
    try:
        if not select.select([worker.stdout], [], [], 5)[0]:
            raise TimeoutError("Mali worker did not become ready")
        if worker.stdout.readline().strip() != "READY":
            raise RuntimeError("Mali worker stopped before becoming ready")
        command = ["/system/bin/sh", str(package / "run_on_phone.sh"), "--client"]
        if args.abi:
            probe = package / "abi_smoke"
            probe.chmod(0o700)
            command = [
                os.environ["MALI_GLIBC_LOADER"], "--library-path",
                os.environ["MALI_GLIBC_LIBRARIES"], str(probe),
                str(package / "jax_plugins/mali/libpjrt_mali.so"),
            ]
        subprocess.run(command, check=True, timeout=90)
        if worker.wait(timeout=5) != 0:
            raise RuntimeError("Mali worker did not shut down cleanly")
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
        worker.stdout.close()


if __name__ == "__main__":
    main()
