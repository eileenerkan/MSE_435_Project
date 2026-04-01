from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from docx import Document

from clinic_scheduler.data_loader import (
    assigned_room_for_time,
    build_historical_schedule,
    get_blocked_periods,
    load_appointments,
    load_room_assignments,
)


class DataLoaderTests(unittest.TestCase):
    def test_load_appointments_and_build_historical_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "appointments.csv"
            pd.DataFrame(
                [
                    {
                        "Patient Id": "P1",
                        "Appt Date": "03-31-2026",
                        "Primary Provider": "Dr A",
                        "ApptStatusSingleView": "Complete",
                        "No Show Appts": "N",
                        "Appt Time": "09:10:00",
                        "Appt Duration": 20,
                        "Appt Type": "Follow Up",
                        "Cancelled Appts": "N",
                        "Deleted Appts": "N",
                    },
                    {
                        "Patient Id": "P2",
                        "Appt Date": "03-31-2026",
                        "Primary Provider": "Dr A",
                        "ApptStatusSingleView": "Complete",
                        "No Show Appts": "Y",
                        "Appt Time": "13:05:00",
                        "Appt Duration": 15,
                        "Appt Type": "Consult",
                        "Cancelled Appts": "N",
                        "Deleted Appts": "N",
                    },
                    {
                        "Patient Id": "P3",
                        "Appt Date": "03-31-2026",
                        "Primary Provider": "Dr B",
                        "ApptStatusSingleView": "Cancelled",
                        "No Show Appts": "N",
                        "Appt Time": "10:00:00",
                        "Appt Duration": 30,
                        "Appt Type": "Consult",
                        "Cancelled Appts": "Y",
                        "Deleted Appts": "N",
                    },
                ]
            ).to_csv(csv_path, index=False)

            appointments_df = load_appointments(str(csv_path))

            self.assertEqual(appointments_df["patient_id"].tolist(), ["P1", "P2"])
            self.assertEqual(appointments_df["start_min"].tolist(), [550, 785])
            self.assertEqual(appointments_df["slots"].tolist(), [4, 3])
            self.assertEqual(appointments_df["no_show"].tolist(), [False, True])
            self.assertEqual(appointments_df["date_str"].tolist(), ["2026-03-31", "2026-03-31"])
            self.assertEqual(appointments_df["appointment_id"].nunique(), 2)

            room_assignments = {
                "Dr A": {
                    "Tuesday": {"am_room": 2, "pm_room": 5, "available": True, "status": "SPLIT"},
                }
            }
            schedule_df = build_historical_schedule(appointments_df, room_assignments)

            self.assertEqual(schedule_df["room"].tolist(), [2, 5])
            self.assertEqual(schedule_df["day"].tolist(), ["Tuesday", "Tuesday"])
            self.assertEqual(schedule_df["date"].tolist(), ["2026-03-31", "2026-03-31"])
            self.assertEqual(schedule_df["end_min"].tolist(), [570, 800])
            self.assertEqual(schedule_df["occupied_end_min"].tolist(), [570, 800])
            self.assertEqual(schedule_df["buffered_slots"].tolist(), [4, 3])
            self.assertEqual(schedule_df["is_phantom"].tolist(), [False, False])

    def test_load_room_assignments_and_room_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            docx_path = Path(tmp_dir) / "assignments.docx"
            document = Document()
            table = document.add_table(rows=1, cols=6)
            table.rows[0].cells[0].text = "Provider"
            for idx, day_name in enumerate(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], start=1):
                table.rows[0].cells[idx].text = day_name

            row = table.add_row().cells
            row[0].text = "Dr Split"
            row[1].text = "ROOM 1"
            row[2].text = "ROOM 2 (AM) / ROOM 4 (PM)"
            row[3].text = "NO ROOM AVAILABLE"
            row[4].text = "CLOSED"
            row[5].text = "N/A"
            document.save(docx_path)

            assignments = load_room_assignments(str(docx_path))

            self.assertEqual(assignments["Dr Split"]["Monday"]["status"], "FULL_DAY")
            self.assertEqual(assignments["Dr Split"]["Tuesday"]["am_room"], 2)
            self.assertEqual(assignments["Dr Split"]["Tuesday"]["pm_room"], 4)
            self.assertEqual(assignments["Dr Split"]["Wednesday"]["status"], "NO ROOM AVAILABLE")
            self.assertFalse(assignments["Dr Split"]["Thursday"]["available"])
            self.assertEqual(assignments["Dr Split"]["Friday"]["status"], "N/A")

            self.assertEqual(assigned_room_for_time(assignments, "Dr Split", "Tuesday", 600), 2)
            self.assertEqual(assigned_room_for_time(assignments, "Dr Split", "Tuesday", 780), 4)
            self.assertIsNone(assigned_room_for_time(assignments, "Missing", "Tuesday", 780))

    def test_blocked_periods_are_opt_in_for_solver_policies(self) -> None:
        self.assertEqual(get_blocked_periods(0, {}), [])
        self.assertTrue(get_blocked_periods(0, {"respect_blocked_periods": True}))


if __name__ == "__main__":
    unittest.main()
