"""Capture Android Perfetto power counters and export device-time CSV files."""

import argparse
import bisect
import csv
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
import uuid


def fields(data):
    """Iterate wire fields used by TracePacket, BatteryCounters and PowerRails."""
    def varint(offset):
        value = 0
        for shift in range(0, 70, 7):
            if offset >= len(data):
                break
            byte = data[offset]
            offset += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value, offset
        raise ValueError("Invalid protobuf varint")

    offset = 0
    while offset < len(data):
        tag, offset = varint(offset)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = varint(offset)
        elif wire in (1, 2, 5):
            if wire == 2:
                length, offset = varint(offset)
            else:
                length = 8 if wire == 1 else 4
            end = offset + length
            if end > len(data):
                raise ValueError("Truncated Perfetto field")
            value, offset = data[offset:end], end
        else:
            raise ValueError(f"Unsupported Perfetto wire type {wire}")
        yield number, value


def power_counters(trace):
    """Read the power messages emitted by the capture config below."""
    battery, descriptors, rails = {}, {}, {}
    for number, packet in fields(trace):
        if number != 1:
            continue
        entries = dict(fields(packet))
        if 38 in entries and 8 in entries:
            if entries.get(58, 6) != 6:
                raise ValueError("Battery timestamps must use CLOCK_BOOTTIME")
            counters = dict(fields(entries[38]))
            if 3 in counters:
                current = counters[3]
                if current >= 2**63:
                    current -= 2**64
                battery[entries[8] / 1e9] = (current, counters.get(7))
        if 40 in entries:
            readings = list(fields(entries[40]))
            session = dict(readings).get(3, 0)
            for kind, value in readings:
                if kind == 1:
                    item = dict(fields(value))
                    if 1 in item and 2 in item:
                        descriptors[session, item[1]] = item[2].decode("utf-8")
                elif kind == 2:
                    item = dict(fields(value))
                    if all(key in item for key in (1, 2, 3)):
                        rails.setdefault((session, item[1]), {})[item[2] / 1000] = item[3] / 1e6
    return battery, [(descriptors.get(key, f"rail {key[1]}"), sorted(samples.items()))
                     for key, samples in sorted(rails.items())]


def battery_snapshot(adb):
    output = subprocess.check_output([*adb, "shell", "cat /proc/uptime; dumpsys battery"],
                                     text=True, timeout=10)
    timestamp = float(output.split()[0])
    supplies = dict(re.findall(r"(AC|USB|Wireless|Dock) powered:\s*(true|false)", output))
    if not {"AC", "USB", "Wireless"} <= supplies.keys():
        raise ValueError("Android did not report its external power state")
    voltage = re.search(r"^\s*voltage:\s*(\d+)\s*$", output, flags=re.M)
    if not voltage:
        raise ValueError("Android did not report a battery voltage")
    return timestamp, 1000 * int(voltage[1]), "true" in supplies.values()


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def export_power(directory, battery, rails, snapshots):
    times = [record[0] for record in snapshots]
    rows = []
    for timestamp, (current, voltage) in sorted(battery.items()):
        right = bisect.bisect_right(times, timestamp)
        if not 0 < right < len(snapshots):
            continue
        before, after = snapshots[right - 1], snapshots[right]
        if voltage is None:
            fraction = (timestamp - before[0]) / (after[0] - before[0])
            voltage = round(before[1] + fraction * (after[1] - before[1]))
        rows.append((f"{timestamp:.6f}", current, voltage, int(before[2] or after[2])))
    write_csv(directory / "battery.csv", ("time_s", "current_ua", "voltage_uv", "powered"), rows)
    summary = {"battery_samples": len(rows), "externally_powered": any(row[2] for row in snapshots),
               "voltage_source": "Perfetto when present, otherwise interpolated dumpsys battery",
               "clock": "Android CLOCK_BOOTTIME in seconds", "rails": []}
    for index, (name, samples) in enumerate(rails):
        filename = f"rail-{index}.csv"
        write_csv(directory / filename, ("time_s", "energy_j"), samples)
        summary["rails"].append({"name": name, "file": filename, "samples": len(samples)})
    (directory / "capture.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def capture(adb, directory, duration):
    trace = directory / "power.pftrace"
    remote = f"/data/misc/perfetto-traces/mali-benchmark-{uuid.uuid4().hex}.pftrace"
    config = f"""buffers {{ size_kb: 4096 fill_policy: RING_BUFFER }}
duration_ms: {duration * 1000}
data_sources {{ config {{ name: "android.power" android_power_config {{
  battery_poll_ms: 250 battery_counters: BATTERY_COUNTER_CURRENT collect_power_rails: true
}} }} }}
"""
    snapshots = [battery_snapshot(adb)]
    try:
        with tempfile.TemporaryFile() as config_input:
            config_input.write(config.encode("ascii"))
            config_input.seek(0)
            with subprocess.Popen([*adb, "shell", "-T", "perfetto", "--txt", "-c", "-", "-o", remote],
                                  stdin=config_input) as recorder:
                deadline = time.monotonic() + duration + 15
                try:
                    while recorder.poll() is None:
                        if time.monotonic() > deadline:
                            raise RuntimeError("Perfetto capture timed out")
                        time.sleep(0.75)
                        snapshots.append(battery_snapshot(adb))
                finally:
                    if recorder.poll() is None:
                        recorder.kill()
                if recorder.returncode:
                    raise RuntimeError(f"Perfetto exited with status {recorder.returncode}")
        snapshots.append(battery_snapshot(adb))
        subprocess.run([*adb, "pull", remote, str(trace)], check=True, timeout=30)
    finally:
        subprocess.run([*adb, "shell", "rm", "-f", remote], check=False, timeout=10)
    export_power(directory, *power_counters(trace.read_bytes()), snapshots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="Create a new directory for this capture")
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--serial", help="ADB serial if multiple devices are connected")
    args = parser.parse_args()
    if args.seconds < 2:
        parser.error("--seconds must be at least 2")
    args.out.mkdir(parents=True)
    capture(["adb", "-s", args.serial] if args.serial else ["adb"], args.out, args.seconds)


if __name__ == "__main__":
    main()
