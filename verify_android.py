"""Run the two compiled JAX fixtures with the Android Vulkan runtime."""

import argparse
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--predict", type=Path, required=True)
    parser.add_argument("--train-step", type=Path, required=True)
    parser.add_argument("--serial", help="ADB serial if multiple devices are attached")
    args = parser.parse_args()
    for path in (args.runner, args.predict, args.train_step):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    adb_command = ["adb"]
    if args.serial:
        adb_command.extend(["-s", args.serial])

    def adb(*arguments, **options):
        return subprocess.run([*adb_command, *arguments], text=True, timeout=60, **options)

    def shell(*arguments, **options):
        return adb("shell", "-T", shlex.join(arguments), **options)

    remote = shell(
        "mktemp", "-d", "/data/local/tmp/jax-mali.XXXXXX", capture_output=True, check=True
    ).stdout.strip()
    runner = f"{remote}/iree-run-module"
    predict = f"{remote}/predict.vmfb"
    train_step = f"{remote}/train_step.vmfb"
    fixed_inputs = ("--input=2xf32=1 2", "--input=2x2xf32=1 2 3 4")
    try:
        for local, destination in ((args.runner, runner), (args.predict, predict),
                                   (args.train_step, train_step)):
            adb("push", str(local), destination, check=True)
        shell("chmod", "700", runner, check=True)
        shell(runner, "--list_devices=vulkan", check=True)
        shell(runner, "--device=vulkan", f"--module={predict}", "--function=main",
              *fixed_inputs, "--expected_output=2xf32=5 11", check=True)
        shell(runner, "--device=vulkan", f"--module={train_step}", "--function=main",
              *fixed_inputs, "--input=2xf32=3 7",
              "--expected_output=2xf32=-0.75 -0.5", check=True)
        print("PASS: Vulkan forward and one gradient update")
    finally:
        shell("rm", "-f", runner, predict, train_step)
        shell("rmdir", remote)


if __name__ == "__main__":
    main()
