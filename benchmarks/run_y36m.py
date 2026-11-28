"""Run the Qwen and PJRT performance campaigns sequentially on a Y36m."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RANKER_REVISION = "4d42700a99165d43917d27555ffc613c302bff36"
IREE_REVISION = "e4a3b0405d7d23554da26403658d0e8c3c5ecf25"


def run(arguments, *, cwd=ROOT):
    subprocess.run(list(map(str, arguments)), cwd=cwd, check=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def revision(repository):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()


def qwen(output, repository):
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repository):
        raise RuntimeError("Commit the Qwen source changes before benchmarking that revision")
    output.mkdir(parents=True)
    python = sys.executable
    run([python, "build_android.py"], cwd=repository)
    run([python, "run.py", "kernel", "--out", output], cwd=repository)
    run([python, "run.py", "verify", "--tokens", 24, "--both-soa", "--out", output / "verified"],
        cwd=repository)
    shapes = [("shape-profile/aligned", "--both-soa", 24, True),
              ("shape-profile/long-aligned", "--both-soa", 96, False)]
    layouts = [("weight-layout/1-raw", ""), ("weight-layout/2-soa", "--weight-soa"),
               ("weight-layout/3-soa", "--weight-soa"), ("weight-layout/4-raw", ""),
               ("activation-layout/1-weights", "--weight-soa"), ("activation-layout/2-both", "--both-soa"),
               ("activation-layout/3-both", "--both-soa"), ("activation-layout/4-weights", "--weight-soa")]
    for location, layout, tokens, profile in shapes + [(name, flag, 24, True) for name, flag in layouts]:
        command = [python, "run.py", "compare", "--tokens", tokens, "--out", output / location]
        if profile:
            command.append("--profile-shaders")
        if layout:
            command.append(layout)
        run(command, cwd=repository)
    write_json(output / "SOURCE.json", {
        "repository": subprocess.check_output(["git", "remote", "get-url", "origin"],
                                             cwd=repository, text=True).strip(),
        "revision": revision(repository), "device": "vivo Y36m (Mali-G57 MC2)",
        "model": "Qwen2.5-0.5B-Instruct Q4_0",
        "runner": "benchmarks/run_y36m.py --suite qwen",
        "files": sorted(path.relative_to(output).as_posix() for path in output.rglob("*")
                        if path.suffix in (".json", ".csv")),
    })


def phone_run(output, remote, package, name, arguments, profile="baseline", *, resume=False):
    path = output / f"{name}.log"
    if resume and path.is_file():
        previous = path.read_text(encoding="utf-8")
        if previous.endswith("\nEXIT 0\n"):
            print(f"USE {name} ({profile})", flush=True)
            return previous
    spec = "" if profile == "baseline" else f"{remote}/{profile}.mlir"
    command = ["env", f"MALI_IREE_TUNING_SPEC={spec}", "sh", f"{remote}/{package}/run_on_phone.sh", *arguments]
    print(f"RUN {name} ({profile})", flush=True)
    lines = []
    with subprocess.Popen(["adb", "shell", "-T", shlex.join(command)], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, encoding="utf-8", errors="replace") as process:
        with path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                lines.append(line)
            code = process.wait()
            log.write(f"\nEXIT {code}\n")
        if code != 0:
            raise RuntimeError(f"Phone run {name} failed, see {path}")
    return "".join(lines)


def probe_record(index, profile, text):
    forward = {dtype: float(duration) for dtype, duration in
               re.findall(r"storage=(float32|float16), activation_bytes=\d+, two_layers_ms=([\d.]+)", text)}
    backward = float(re.search(r"backward_two_matmuls_ms=([\d.]+)", text)[1])
    return {"run": index, "profile": profile, "forward_fp32_activation_ms": forward["float32"],
            "forward_fp16_activation_ms": forward["float16"], "backward_ms": backward}


def training_record(profile, text):
    block_seconds = [float(value) for value in re.findall(r"^block=\d+/12, train_s=([\d.]+)", text, re.M)]
    if len(block_seconds) != 12:
        raise RuntimeError(f"Expected twelve training blocks for {profile}")
    metrics = re.search(r"total_train_s=([\d.]+), pairs_per_s=(\d+), "
                        r"first_window_pairs_per_s=(\d+), last_window_pairs_per_s=(\d+), "
                        r"eval_loss=([\d.]+)->([\d.]+), accuracy=([\d.]+), ndcg@5=([\d.]+)", text)
    return {"profile": profile, "compile_and_load_s": float(re.search(r"compile_and_load_s=([\d.]+)", text)[1]),
            "block_seconds": block_seconds, "total_train_s": float(metrics[1]), "pairs_per_s": int(metrics[2]),
            "first_window_pairs_per_s": int(metrics[3]), "last_window_pairs_per_s": int(metrics[4]),
            "eval_loss_before": float(metrics[5]), "eval_loss_after": float(metrics[6]),
            "accuracy": float(metrics[7]), "ndcg_at_5": float(metrics[8])}


def gemm_record(profile, text):
    blocks = [float(value) for value in re.findall(r"^gemm_block=\d+/8, seconds=([\d.]+)", text, re.M)]
    metrics = re.search(r"^total_gemm_s=([\d.]+), effective_gflops=([\d.]+)", text, re.M)
    if len(blocks) != 8 or metrics is None:
        raise RuntimeError(f"Incomplete GEMM timings for {profile}")
    return {"profile": profile, "block_seconds": blocks, "total_gemm_s": float(metrics[1]),
            "effective_gflops": float(metrics[2])}


def compute_controls(output, remote, package, *, resume):
    controls = {}
    for case, field, parse in (("fp32", "fp32_training", training_record), ("gemm", "gemm", gemm_record)):
        runs = []
        for platform, profile in (("cpu", "baseline"), ("mali", "reuse-gradients")):
            report = phone_run(output, remote, package, f"{case}-{platform}", [platform, case],
                               profile, resume=resume)
            runs.append({"platform": platform, **parse(profile, report)})
        controls[field] = runs
    return controls


def record_valhall_inputs(output, remote, inputs):
    manifest = output / "inputs.json"
    record = {
        "device_serial": subprocess.check_output(["adb", "get-serialno"], text=True).strip(),
        "device_workspace": remote,
        "sha256": {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in inputs},
    }
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding="utf-8")) != record:
            raise RuntimeError("The Valhall inputs changed. Use a new output directory.")
    elif any(output.glob("*.log")):
        raise RuntimeError("The existing Valhall logs lack an input manifest. Use a new output directory.")
    else:
        write_json(manifest, record)


def valhall(output, remote, zig, revisions, *, resume):
    output.mkdir(parents=True, exist_ok=resume)
    profiles = output / "profiles"
    python = sys.executable
    run([python, "benchmarks/mali_tuning.py", profiles])
    selected = ROOT / "benchmarks/data/valhall-y36m/reuse-gradients.mlir"
    spec = selected.read_bytes().replace(b"\r\n", b"\n")
    (profiles / selected.name).write_bytes(spec)
    package = output / "pjrt-valhall"
    run([python, "pjrt/build.py", "--zig", zig, "--output", package])
    training = output / "pjrt-valhall-training"
    shutil.copytree(package, training, dirs_exist_ok=resume)
    shutil.copyfile(package / "smoke.py", training / "pjrt_smoke.py")
    shutil.copyfile(ROOT / "benchmarks/valhall_ranker.py", training / "smoke.py")
    candidates = ["baseline", "light", "balanced", "reuse", "shared"]
    specs = [profiles / f"{name}.mlir" for name in (*candidates[1:], "reuse-gradients")]
    record_valhall_inputs(output, remote, [*(path for path in package.rglob("*") if path.is_file()),
                                          training / "smoke.py", *specs])
    run(["adb", "push", package, training, *specs, remote + "/"])

    order = [*candidates, *reversed(candidates), "reuse-gradients"]
    probes = [probe_record(index, profile, phone_run(output, remote, package.name, f"probe-{index:02d}",
                                                     ["--valhall"], profile, resume=resume))
              for index, profile in enumerate(order, 1)]
    runs = []
    for profile in ("cpu", "baseline", "reuse-gradients", "reuse"):
        platform = "cpu" if profile == "cpu" else "mali"
        config = "baseline" if profile == "cpu" else profile
        text = phone_run(output, remote, training.name, f"train-{profile}", [platform], config, resume=resume)
        runs.append(training_record(profile, text))

    write_json(output / "results.json", {
        "device": {"model": "vivo Y36m (V2318A)", "soc": "MT6833", "gpu": "Mali-G57 MC2", "android": 13},
        "revisions": {**revisions, "archived_ranker": RANKER_REVISION,
                      "iree": IREE_REVISION, "jax": "0.6.1"},
        "bundle_sha256": {
            "pjrt_library": hashlib.sha256((package / "jax_plugins/mali/libpjrt_mali.so").read_bytes()).hexdigest(),
            "training_fixture": hashlib.sha256((training / "smoke.py").read_bytes()).hexdigest(),
        },
        "probe": {"measured_calls_per_trial": 7, "trial_metric": "median_ms", "trials": probes},
        "training": {"precision": "mixed", "layers": [32, 1024, 1024, 1], "batch": 4096,
                     "blocks": 12, "updates": 96, "processed_pairs": 393216, "runs": runs},
        "reuse_gradients": {"source_profile": "reuse", "selectors": ["input_gradient", "weight_gradient"],
                            "filename": selected.name, "sha256": hashlib.sha256(spec).hexdigest()},
    })
    write_json(output / "compute-controls.json", {
        **revisions,
        "fixture_origin": RANKER_REVISION,
        "fixture_sha256": hashlib.sha256((training / "smoke.py").read_bytes()).hexdigest(),
        **compute_controls(output, remote, training.name, resume=resume),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--suite", choices=("qwen", "valhall", "power", "all"), default="all")
    parser.add_argument("--qwen-repo", type=Path, default=ROOT.parent / "mali-q4-decode")
    parser.add_argument("--phone-root", default="/data/local/tmp/jax-mali-train.qCUUWZ")
    parser.add_argument("--zig", type=Path, default=ROOT / "build/pjrt-deps/zig-x86_64-windows-0.15.2/zig.exe")
    parser.add_argument("--resume", action="store_true", help="Resume completed Valhall runs in this output directory")
    args = parser.parse_args()
    if args.resume and args.suite != "valhall":
        parser.error("--resume requires --suite valhall")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.suite in ("qwen", "all"):
        qwen(output / "qwen", args.qwen_repo.resolve())
    if args.suite in ("valhall", "all"):
        optimization_revision = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", "pjrt/executable.c"], cwd=ROOT, text=True).strip()
        revisions = {"pjrt": optimization_revision, "pjrt_build_source": revision(ROOT)}
        valhall(output / "valhall", args.phone_root, args.zig.resolve(), revisions, resume=args.resume)
    if args.suite == "power":
        run([sys.executable, "benchmarks/capture_power.py", output / "power"])


if __name__ == "__main__":
    main()
