"""Stage matching Android debug binaries and start an ADB-forwarded LLDB server."""

import argparse
import os
from pathlib import Path
from queue import Empty, Queue
import re
import shlex
import subprocess
import sys
from threading import Thread


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "build/pjrt-vscode"
SYMBOLS = ROOT / "build/pjrt-vscode-symbols"
PORT = 49185
PHONE = os.environ.get("JAX_MALI_PHONE_DIR", "/data/local/tmp/jax-mali-train.qCUUWZ")
REMOTE_PACKAGE = f"{PHONE}/{PACKAGE.name}"
LOADER = "ld-linux-aarch64.so.1"
WORKER_SOCKET = "mali-vscode-worker"
PLUGIN = "jax_plugins/mali/libpjrt_mali.so"
# LLDB does not discover this plugin through the manually invoked glibc loader.
# Preload it without initializing PJRT, and report the mapping before attaching.
PLUGIN_BOOTSTRAP = """
import ctypes
from pathlib import Path
import runpy
import sys
import time

library, base_file, gate, smoke = sys.argv[1:]
plugin = ctypes.CDLL(library)
for entry in Path('/proc/self/maps').read_text().splitlines():
    addresses, _, offset, _, _, path = (entry.split(maxsplit=5) + [''])[:6]
    if path == library and int(offset, 16) == 0:
        Path(base_file).write_text('0x' + addresses.split('-')[0] + '\\n')
        break
else:
    raise RuntimeError('Could not find the PJRT plugin mapping')
while not Path(gate).exists():
    time.sleep(0.05)
sys.argv = [smoke, '--client']
runpy.run_path(smoke, run_name='__main__')
"""


def adb(*args):
    return subprocess.run(["adb", *args], check=True, cwd=ROOT)


def prepare():
    zig = ROOT / "build/pjrt-deps/zig-x86_64-windows-0.15.2/zig.exe"
    subprocess.run([sys.executable, str(ROOT / "pjrt/build.py"), "--debug", "--zig", str(zig),
                    "--output", str(PACKAGE)], check=True, cwd=ROOT)
    clang = ROOT.parent / ".gpu-android-tools/android-ndk-r30/toolchains/llvm/prebuilt/windows-x86_64/lib/clang"
    servers = sorted(clang.glob("*/lib/linux/aarch64/lldb-server"), reverse=True)
    if not servers:
        raise RuntimeError(f"Could not find the arm64 NDK lldb-server under {clang}")
    adb("push", str(servers[0]), f"{PHONE}/lldb-server")
    adb("shell", shlex.join(["mkdir", "-p", REMOTE_PACKAGE]))
    package_files = ("mali_worker", "run_on_phone.sh", "smoke.py", "training_fixture.py", "jax_plugins")
    adb("push", *(str(PACKAGE / name) for name in package_files), f"{REMOTE_PACKAGE}/")
    adb("shell", shlex.join(["chmod", "700", f"{PHONE}/lldb-server", f"{REMOTE_PACKAGE}/mali_worker"]))


def read_port(process, ready, plugin):
    ready.put(process.stdout.readline().decode(errors="replace").strip())
    if plugin:
        ready.put(process.stdout.readline().decode(errors="replace").strip())
    data = bytearray()
    while byte := process.stdout.read(1):
        if byte == b"\0":
            break
        data.extend(byte)
    ready.put(data.decode(errors="replace").strip())


def run_server(mode):
    port = f"tcp:{PORT}"
    adb("forward", "--no-rebind", port, port)
    try:
        prepare()
        plugin = mode == "plugin"
        command = (plugin_command() if plugin else
                   f"start_worker {WORKER_SOCKET}\nstart_server \"$worker_pid\"")
        command = remote_supervisor() + command
        with subprocess.Popen(["adb", "shell", "-T", command], stdout=subprocess.PIPE,
                              stdin=subprocess.DEVNULL, cwd=ROOT) as process:
            supervisor_pid = None
            try:
                ready = Queue()
                Thread(target=read_port, args=(process, ready, plugin), daemon=True).start()
                try:
                    supervisor = ready.get(timeout=10)
                    if supervisor.startswith("MALI_PID=") and supervisor[9:].isdecimal():
                        supervisor_pid = supervisor[9:]
                    else:
                        raise RuntimeError(f"Could not start Android debug supervisor: {supervisor!r}")
                    if plugin:
                        mapping = ready.get(timeout=20)
                        match = re.fullmatch(r"MALI_PLUGIN_BASE=(0x[0-9a-fA-F]+)", mapping)
                        if match is None:
                            raise RuntimeError(f"Could not locate the phone's PJRT plugin: {mapping!r}")
                        module = (PACKAGE / PLUGIN).as_posix()
                        (SYMBOLS / "plugin-symbols.lldb").write_text(
                            f'target modules load --file "{module}" --slide {match[1]}\n', encoding="utf-8")
                    actual = ready.get(timeout=20)
                except Empty as error:
                    raise RuntimeError("Timed out waiting for Android lldb-server") from error
                if actual != str(PORT):
                    raise RuntimeError(f"Android lldb-server did not report port {PORT}: {actual!r}")
                print("MALI_DEBUG_READY", flush=True)
                for line in process.stdout:
                    print(line.decode(errors="replace").rstrip(), flush=True)
                return process.wait()
            finally:
                if process.poll() is None:
                    if supervisor_pid is not None:
                        subprocess.run(["adb", "shell", "kill", "-TERM", supervisor_pid], cwd=ROOT,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.terminate()
    finally:
        subprocess.run(["adb", "forward", "--remove", port], cwd=ROOT)


def plugin_command():
    glibc = f"{PHONE}/glibc/usr/lib/aarch64-linux-gnu"
    python = f"{PHONE}/python"
    SYMBOLS.mkdir(parents=True, exist_ok=True)
    adb("pull", f"{glibc}/{LOADER}", str(SYMBOLS / LOADER))
    variables = {
        "PYTHONHOME": python,
        "PYTHONPATH": f"{REMOTE_PACKAGE}:{PHONE}/site-packages",
        "TMPDIR": f"{PHONE}/tmp",
        "JAX_PLATFORMS": "mali",
        "MALI_PJRT_SOCKET": "mali-vscode-plugin",
        "MALI_GLIBC_LOADER": f"{glibc}/{LOADER}",
        "MALI_GLIBC_LIBRARIES": f"{glibc}:{python}/lib",
        "MALI_IREE_COMPILER": f"{PHONE}/site-packages/iree/compiler/_mlir_libs/iree-compile",
    }
    exports = "\n".join(f"export {name}={shlex.quote(value)}" for name, value in variables.items())
    # Read READY from the worker itself. Its stdout goes to a private file so
    # the debugger's port announcement is the first message on the ADB pipe.
    return f"""{exports}
start_worker "$MALI_PJRT_SOCKET"
"$MALI_GLIBC_LOADER" --library-path "$MALI_GLIBC_LIBRARIES" \
    {shlex.join([f'{python}/bin/python3.11', '-c', PLUGIN_BOOTSTRAP, f'{REMOTE_PACKAGE}/{PLUGIN}'])} \
    "$run_dir/base" "$run_dir/gate" {shlex.quote(f'{REMOTE_PACKAGE}/smoke.py')} &
python_pid=$!
while [ ! -s "$run_dir/base" ] && kill -0 "$python_pid" 2>/dev/null; do sleep 0.1; done
printf 'MALI_PLUGIN_BASE='
cat "$run_dir/base"
start_server "$python_pid"
"""


def remote_supervisor():
    server = shlex.join([f"{PHONE}/lldb-server", "gdbserver"])
    return f"""set -eu
run_dir=$(mktemp -d {shlex.quote(f'{PHONE}/tmp/mali-vscode.XXXXXX')})
cleanup() {{
    if [ -n "${{server_pid:-}}" ]; then kill "$server_pid" 2>/dev/null || true; fi
    for pid in "${{python_pid:-}}" "${{worker_pid:-}}"; do
        if [ -n "$pid" ]; then kill -KILL "$pid" 2>/dev/null || true; fi
    done
    rm -f "$run_dir/port" "$run_dir/worker.log" "$run_dir/server.log" "$run_dir/base" "$run_dir/gate"
    rmdir "$run_dir"
}}
trap cleanup EXIT
trap 'exit 143' TERM
printf 'MALI_PID=%s\\n' "$$"
start_worker() {{
    {shlex.quote(f'{REMOTE_PACKAGE}/mali_worker')} "$1" >"$run_dir/worker.log" &
    worker_pid=$!
    while ! grep -q '^READY$' "$run_dir/worker.log"; do
        if ! kill -0 "$worker_pid" 2>/dev/null; then cat "$run_dir/worker.log" >&2; return 1; fi
        sleep 0.1
    done
}}
start_server() {{
    {server} --pipe 3 --attach "$1" 127.0.0.1:{PORT} 3>"$run_dir/port" >"$run_dir/server.log" &
    server_pid=$!
    while [ ! -s "$run_dir/port" ] && kill -0 "$server_pid" 2>/dev/null; do sleep 0.1; done
    if [ -s "$run_dir/port" ]; then
        if [ -n "${{python_pid:-}}" ]; then
            while ! grep -Eq '^TracerPid:[[:space:]]+[1-9]' "/proc/$python_pid/status"; do
                kill -0 "$server_pid" 2>/dev/null || return 1
                sleep 0.1
            done
            touch "$run_dir/gate"
        fi
        cat "$run_dir/port"
    fi
    result=0
    wait "$server_pid" || result=$?
    cat "$run_dir/server.log"
    return "$result"
}}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plugin", "worker", "client"))
    mode = parser.parse_args().mode
    if mode == "client":
        adb("shell", "-T", shlex.join(["env", f"MALI_PJRT_SOCKET={WORKER_SOCKET}", "sh",
                                        f"{REMOTE_PACKAGE}/run_on_phone.sh", "--client"]))
        return 0

    print("MALI_DEBUG_START", flush=True)
    return run_server(mode)


if __name__ == "__main__":
    raise SystemExit(main())
