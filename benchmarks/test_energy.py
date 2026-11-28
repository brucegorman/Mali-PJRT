import tempfile
import unittest
from pathlib import Path

from plot_energy import measure


class EnergyMeasurementTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.run_spec = {"label": "test", "csv": "samples.csv", "window_s": [0.5, 2.5], "units": 3}

    def write(self, name, text):
        (self.root / name).write_text(text, encoding="utf-8")

    def test_clips_power_and_integrates_at_progress_events(self):
        self.write("samples.csv", "time_s,power_w\n0,1\n1,3\n2,5\n3,7\n")
        self.write("progress.csv", "time_s,completed_units\n0.5,0\n1.25,1\n2.5,3\n")
        self.run_spec["progress_csv"] = "progress.csv"
        summary, _, _, _, efficiency = measure(self.run_spec, self.root, "meter", 1)
        self.assertAlmostEqual(summary["joules"], 8)
        self.assertAlmostEqual(summary["units_per_joule"], 3 / 8)
        before, after = efficiency[1][efficiency[0] == 0.75]
        self.assertEqual(before, 0)
        self.assertAlmostEqual(after, 1 / 2.0625)

    def test_clips_cumulative_rail(self):
        self.write("samples.csv", "time_s,energy_j\n0,100\n1,106\n2,113\n3,121\n")
        self.run_spec["window_s"] = [0.5, 1.5]
        summary, *_ = measure(self.run_spec, self.root, "rail", 1)
        self.assertAlmostEqual(summary["joules"], 6.5)

    def test_uses_battery_units_and_rejects_external_power(self):
        data = "time_s,current_ua,voltage_uv,powered\n" + "".join(
            f"{time},-500000,4000000,0\n" for time in range(4)
        )
        self.write("samples.csv", data)
        summary, *_ = measure(self.run_spec, self.root, "battery", 1)
        self.assertAlmostEqual(summary["joules"], 4)
        self.write("samples.csv", data.replace("2,-500000,4000000,0", "2,-500000,4000000,1"))
        with self.assertRaisesRegex(ValueError, "unplugged"):
            measure(self.run_spec, self.root, "battery", 1)

    def test_rejects_counter_reset_and_sample_gaps(self):
        self.write("samples.csv", "time_s,energy_j\n0,100\n1,105\n2,3\n3,5\n")
        with self.assertRaisesRegex(ValueError, "reset"):
            measure(self.run_spec, self.root, "rail", 1)
        with self.assertRaisesRegex(ValueError, "gap"):
            measure(self.run_spec, self.root, "rail", 0.5)
