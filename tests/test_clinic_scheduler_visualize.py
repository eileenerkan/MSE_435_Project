from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib
import pandas as pd

from clinic_scheduler.visualize import (
    plot_gantt_by_provider,
    plot_gantt_by_room,
    plot_historical_baseline,
    plot_room_utilization_heatmap,
)

matplotlib.use("Agg")


class VisualizeTests(unittest.TestCase):
    def test_plotting_helpers_generate_baseline_outputs(self) -> None:
        appointments_df = pd.DataFrame(
            [
                {
                    "provider": "Dr A",
                    "date": pd.Timestamp("2026-03-30"),
                    "patient_id": "P1",
                    "appointment_id": "appt-1",
                    "start_min": 570,
                    "duration": 20,
                    "slots": 4,
                    "day_of_week": 0,
                },
                {
                    "provider": "Dr A",
                    "date": pd.Timestamp("2026-03-30"),
                    "patient_id": "P2",
                    "appointment_id": "appt-2",
                    "start_min": 780,
                    "duration": 15,
                    "slots": 3,
                    "day_of_week": 0,
                },
                {
                    "provider": "Dr Missing",
                    "date": pd.Timestamp("2026-03-30"),
                    "patient_id": "P3",
                    "appointment_id": "appt-3",
                    "start_min": 840,
                    "duration": 10,
                    "slots": 2,
                    "day_of_week": 0,
                },
            ]
        )
        room_assignments = {
            "Dr A": {
                "Monday": {"am_room": 3, "pm_room": 8, "available": True, "status": "SPLIT"},
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            baseline_dir = Path(tmp_dir) / "baseline"
            schedule_df = plot_historical_baseline(appointments_df, room_assignments, str(baseline_dir))

            self.assertEqual(schedule_df["room"].iloc[0], 3)
            self.assertEqual(schedule_df["room"].iloc[1], 8)
            self.assertTrue(pd.isna(schedule_df["room"].iloc[2]))
            self.assertTrue((baseline_dir / "historical_schedule.csv").exists())
            self.assertTrue((baseline_dir / "historical_gantt_by_provider_monday.png").exists())
            self.assertTrue((baseline_dir / "historical_gantt_by_room_monday.png").exists())
            self.assertTrue((baseline_dir / "historical_room_utilization_heatmap.png").exists())

    def test_individual_plotters_tolerate_missing_room_assignments(self) -> None:
        schedule_df = pd.DataFrame(
            [
                {
                    "provider": "Dr A",
                    "day": "Monday",
                    "date": "2026-03-30",
                    "appointment_id": "appt-1",
                    "patient_id": "P1",
                    "start_min": 570,
                    "end_min": 590,
                    "occupied_end_min": 590,
                    "room": 3,
                    "slots": 4,
                    "buffered_slots": 4,
                    "is_phantom": False,
                },
                {
                    "provider": "Dr Missing",
                    "day": "Monday",
                    "date": "2026-03-30",
                    "appointment_id": "appt-2",
                    "patient_id": "P2",
                    "start_min": 600,
                    "end_min": 610,
                    "occupied_end_min": 610,
                    "room": None,
                    "slots": 2,
                    "buffered_slots": 2,
                    "is_phantom": False,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plot_gantt_by_provider(schedule_df, "Monday", "Provider View", str(tmp_path / "provider.png"))
            plot_gantt_by_room(schedule_df, "Monday", "Room View", str(tmp_path / "room.png"))
            plot_room_utilization_heatmap(schedule_df, str(tmp_path / "heatmap.png"))

            self.assertTrue((tmp_path / "provider.png").exists())
            self.assertTrue((tmp_path / "room.png").exists())
            self.assertTrue((tmp_path / "heatmap.png").exists())


if __name__ == "__main__":
    unittest.main()
