"""Verify Mali device enumeration and PJRT buffer round trips on Android."""

import argparse
import os
from pathlib import Path
import select
import subprocess
import uuid

def check_device(missing_worker):
    import jax
    import numpy as np

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

    try:
        jax.jit(lambda x: x + x).lower(jax.ShapeDtypeStruct((2,), np.float32)).compile()
    except RuntimeError as error:
        if "Mali PJRT compilation is unavailable" not in str(error):
            raise
        print("PASS: unavailable compilation returned a PJRT error", flush=True)
    else:
        raise RuntimeError("Compilation unexpectedly succeeded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missing-worker", action="store_true")
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
        subprocess.run(
            ["/system/bin/sh", str(package / "run_on_phone.sh"), "--client"],
            check=True, timeout=30,
        )
        if worker.wait(timeout=5) != 0:
            raise RuntimeError("Mali worker did not shut down cleanly")
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
        worker.stdout.close()


if __name__ == "__main__":
    main()
