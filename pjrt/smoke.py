"""Verify Mali PJRT device discovery, buffers, and JIT training on Android."""

import argparse
import os
from pathlib import Path
import select
import signal
import subprocess
import tempfile
import time
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


def check_cache(path, write):
    import jax
    import numpy as np
    from training_fixture import train_eight_steps

    jax.config.update("jax_platforms", "mali")
    if not write:
        os.environ.pop("MALI_IREE_COMPILER", None)
    device, = jax.devices("mali")
    client = device.client
    initial = np.array([1, 2], dtype=np.float32)
    target = np.array([3, 7], dtype=np.float32)
    weights, inputs, targets = (
        jax.device_put(data, device) for data in (initial, np.eye(2, dtype=np.float32), target)
    )
    if write:
        compiled = jax.jit(train_eight_steps).lower(weights, inputs, targets).compile()
        blob = client.serialize_executable(compiled.runtime_executable())
        path.write_bytes(blob)
        print(f"PASS: process {os.getpid()} compiled and saved {len(blob)} executable bytes", flush=True)
        return

    blob = path.read_bytes()
    bad_magic = bytes([blob[0] ^ 1]) + blob[1:]
    bad_code = blob[:-1] + bytes([blob[-1] ^ 1])
    bad_driver = bytearray(blob)
    bad_driver[44] ^= 1  # driver_version in the serialized header
    for corrupted, expected in (
        (bad_magic, "Invalid Mali executable image header"),
        (bad_code, "Mali executable image checksum mismatch"),
        (bytes(bad_driver), "targets a different device, driver, or compiler"),
    ):
        try:
            client.deserialize_executable(corrupted, executable_devices=[device])
        except RuntimeError as error:
            if expected not in str(error):
                raise
        else:
            raise RuntimeError(f"Corrupt executable should have been rejected: {expected}")
    print("PASS: invalid header, altered VMFB, and mismatched driver were rejected", flush=True)

    executable = client.deserialize_executable(blob, executable_devices=[device])
    if client.serialize_executable(executable) != blob:
        raise RuntimeError("Executable image changed after reloading")
    for _ in range(2):
        weights, = executable.execute([weights, inputs, targets])
        if weights.device != device:
            raise RuntimeError("Reloaded executable returned weights on the wrong device")
    final = np.asarray(weights)
    np.testing.assert_allclose(final, target + (initial - target) * (7 / 8) ** 16, rtol=1e-6)
    try:
        jax.jit(lambda value: value + 9)(weights)
    except RuntimeError as error:
        if "Set the phone IREE compiler environment" not in str(error):
            raise
    else:
        raise RuntimeError("A fresh JIT unexpectedly reached the compiler")
    print(f"PASS: process {os.getpid()} loaded and ran the cached training twice with no compiler",
          flush=True)
    print(f"cached weights={final.tolist()}", flush=True)


def check_disconnect(worker_pid, worker_stdout):
    import jax
    import numpy as np

    device, = jax.devices("mali")
    x = jax.device_put(np.array([1, 2], dtype=np.float32), device)
    compiled = jax.jit(lambda value: value + value).lower(x).compile()
    os.kill(worker_pid, signal.SIGSTOP)
    status = Path(f"/proc/{worker_pid}/status")
    while "State:\tT" not in status.read_text():
        time.sleep(0.001)
    failed = compiled(x)
    dependent = compiled(failed)
    os.kill(worker_pid, signal.SIGKILL)
    with os.fdopen(worker_stdout, "rb") as stdout:
        if stdout.read():
            raise RuntimeError("Unexpected output after stopping the Mali worker")

    for action, operation in ((failed.block_until_ready, "execute"),
                              (dependent.block_until_ready, "execute"),
                              (lambda: np.asarray(x), "read")):
        try:
            action()
        except RuntimeError as error:
            if "UNAVAILABLE" not in str(error) or f"Mali worker {operation}:" not in str(error):
                raise
        else:
            raise RuntimeError(f"Mali worker {operation} succeeded after disconnect")
    print("PASS: worker exit fails both queued results and subsequent readback", flush=True)


def check_fences(package):
    source = """module {
  func.func public @double(%x: tensor<4xi32>) -> tensor<4xi32> {
    %y = stablehlo.add %x, %x : tensor<4xi32>
    return %y : tensor<4xi32>
  }
  func.func public @add_three(%y: tensor<4xi32>) -> tensor<4xi32> {
    %three = stablehlo.constant dense<3> : tensor<4xi32>
    %z = stablehlo.add %y, %three : tensor<4xi32>
    return %z : tensor<4xi32>
  }
}"""
    with tempfile.TemporaryDirectory(prefix="pjrt-fences-") as temporary:
        mlir = Path(temporary) / "chain.mlir"
        vmfb = Path(temporary) / "chain.vmfb"
        mlir.write_text(source, encoding="utf-8")
        start = time.monotonic()
        subprocess.run([
            os.environ["MALI_GLIBC_LOADER"], "--library-path", os.environ["MALI_GLIBC_LIBRARIES"],
            os.environ["MALI_IREE_COMPILER"], str(mlir), "--iree-input-type=stablehlo",
            "--iree-hal-target-device=vulkan", "--iree-vulkan-target=mali-g57",
            "--iree-execution-model=async-external", "-o", str(vmfb),
        ], check=True, timeout=60)
        print(f"Compiled two coarse-fences exports on the phone in {time.monotonic() - start:.3f}s",
              flush=True)
        return subprocess.run([str(package / "mali_worker"), "--fence-probe", str(vmfb)],
                              timeout=30).returncode


def run_worker(package, command, disconnect=False, async_probe=False):
    socket_name = f"mali-pjrt-{uuid.uuid4().hex}"
    environment = {**os.environ, "MALI_PJRT_SOCKET": socket_name}
    worker = subprocess.Popen([str(package / "mali_worker"), socket_name],
                              stdout=subprocess.PIPE, text=True, env=environment)
    try:
        if not select.select([worker.stdout], [], [], 5)[0]:
            raise TimeoutError("Mali worker did not become ready")
        if worker.stdout.readline().strip() != "READY":
            raise RuntimeError("Mali worker stopped before becoming ready")
        pass_fds = ()
        if disconnect:
            pass_fds = (worker.stdout.fileno(),)
            command = [*command, "--disconnect-client", str(worker.pid), str(pass_fds[0])]
        if async_probe:
            command = [*command, str(worker.pid)]
        subprocess.run(command, check=True, timeout=30 if disconnect else 90,
                       env=environment, pass_fds=pass_fds)
        expected_exit = -signal.SIGKILL if disconnect else 0
        if worker.wait(timeout=5) != expected_exit:
            raise RuntimeError("Mali worker did not shut down cleanly")
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
        worker.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--missing-worker", action="store_true")
    mode.add_argument("--abi", action="store_true", help="Check compile options and output metadata via C")
    mode.add_argument("--async", dest="async_probe", action="store_true",
                      help="Check pending events and queued execution with a paused worker")
    mode.add_argument("--fences", action="store_true", help="Check IREE Vulkan fence chaining")
    mode.add_argument("--cache", action="store_true", help="Save and reload a JIT in separate processes")
    mode.add_argument("--disconnect", action="store_true", help="Check errors after the worker exits")
    mode.add_argument("--disconnect-client", nargs=2, type=int, help=argparse.SUPPRESS)
    mode.add_argument("--client", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--cache-write", type=Path, help=argparse.SUPPRESS)
    mode.add_argument("--cache-read", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.disconnect_client:
        check_disconnect(*args.disconnect_client)
        return
    if args.client:
        check_device(missing_worker=False)
        return
    if args.cache_write or args.cache_read:
        check_cache(args.cache_write or args.cache_read, write=args.cache_write is not None)
        return
    if args.missing_worker:
        os.environ["MALI_PJRT_SOCKET"] = f"mali-pjrt-{uuid.uuid4().hex}"
        check_device(missing_worker=True)
        return

    package = Path(__file__).resolve().parent
    client_command = ["/system/bin/sh", str(package / "run_on_phone.sh")]
    if args.fences:
        raise SystemExit(check_fences(package))
    elif args.disconnect:
        run_worker(package, client_command, disconnect=True)
    elif args.cache:
        with tempfile.TemporaryDirectory(prefix="pjrt-cache-") as temporary:
            path = str(Path(temporary) / "train.malivmfb")
            run_worker(package, [*client_command, "--cache-write", path])
            run_worker(package, [*client_command, "--cache-read", path])
    elif args.abi or args.async_probe:
        probe = package / "abi_smoke"
        probe.chmod(0o700)
        run_worker(package, [
            os.environ["MALI_GLIBC_LOADER"], "--library-path",
            os.environ["MALI_GLIBC_LIBRARIES"], str(probe),
            str(package / "jax_plugins/mali/libpjrt_mali.so"),
        ], async_probe=args.async_probe)
    else:
        run_worker(package, [*client_command, "--client"])


if __name__ == "__main__":
    main()
