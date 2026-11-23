"""Check JAX execution or export the same functions as StableHLO."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


WEIGHTS = np.array([1, 2], dtype=np.float32)
INPUTS = np.array([[1, 2], [3, 4]], dtype=np.float32)
TARGETS = np.array([3, 7], dtype=np.float32)


def predict(weights, inputs):
    return inputs @ weights


def mse(weights, inputs, targets):
    residuals = predict(weights, inputs) - targets
    return jnp.mean(residuals ** 2)


def train_step(weights, inputs, targets):
    gradient = jax.grad(mse)(weights, inputs, targets)
    return weights - 0.125 * gradient


def train_eight_steps(weights, inputs, targets):
    def update(_, current_weights):
        return train_step(current_weights, inputs, targets)

    return jax.lax.fori_loop(0, 8, update, weights)


def run(platform):
    devices = jax.devices(platform)
    if len(devices) != 1:
        raise SystemExit(f"Expected one {platform} device, found {devices}")
    device = devices[0]
    print(f"platform={device.platform} device={device.device_kind}", flush=True)
    weights, inputs, targets = (
        jax.device_put(value, device) for value in (WEIGHTS, INPUTS, TARGETS)
    )
    predictions = jax.jit(predict)(weights, inputs)
    updated = jax.jit(train_step)(weights, inputs, targets)
    for result, expected in ((predictions, [5, 11]), (updated, [-0.75, -0.5])):
        if result.devices() != {device}:
            raise RuntimeError(f"Expected execution on {device}, got {result.devices()}")
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5, atol=1e-6)
    print("PASS: forward and one gradient update")


def export(output):
    output.mkdir(parents=True, exist_ok=True)
    with jax.default_device(jax.devices("cpu")[0]):
        for function, values in (
            (predict, (WEIGHTS, INPUTS)),
            (train_step, (WEIGHTS, INPUTS, TARGETS)),
            (train_eight_steps, (WEIGHTS, INPUTS, TARGETS)),
        ):
            signature = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values]
            module = jax.jit(function).lower(*signature).compiler_ir(dialect="stablehlo")
            path = output / f"{function.__name__}.mlir"
            path.write_text(str(module), encoding="utf-8")
            print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--platform", choices=["cpu", "iree_vulkan"])
    mode.add_argument("--export", type=Path, metavar="DIRECTORY")
    args = parser.parse_args()
    if args.export is None:
        run(args.platform)
    else:
        export(args.export)


if __name__ == "__main__":
    main()
