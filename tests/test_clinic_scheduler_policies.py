from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from clinic_scheduler.policies import policy_a_single_room


class PolicyTests(unittest.TestCase):
    def test_policy_a_assigns_unique_rooms_by_day_using_volume_priority(self) -> None:
        room_assignments = {
            "Busy": {"Monday": {"am_room": 1, "pm_room": 1, "available": True}},
            "Light": {"Monday": {"am_room": 1, "pm_room": 1, "available": True}},
            "Other": {"Monday": {"am_room": 3, "pm_room": 3, "available": True}},
        }
        appointments_df = pd.DataFrame(
            [
                {
                    "provider": "Busy",
                    "date": pd.Timestamp("2025-11-10"),
                },
                {
                    "provider": "Busy",
                    "date": pd.Timestamp("2025-11-10"),
                },
                {
                    "provider": "Light",
                    "date": pd.Timestamp("2025-11-10"),
                },
                {
                    "provider": "Other",
                    "date": pd.Timestamp("2025-11-10"),
                },
            ]
        )
        dist_matrix = np.full((16, 16), 100.0)
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix[0, 1] = 1.0
        dist_matrix[1, 0] = 1.0
        dist_matrix[0, 2] = 2.0
        dist_matrix[2, 0] = 2.0

        policy = policy_a_single_room(room_assignments, appointments_df, dist_matrix)
        fixed_rooms = policy["fixed_room_all_day"]

        self.assertEqual(fixed_rooms[("Busy", "Monday")], 1)
        self.assertEqual(fixed_rooms[("Other", "Monday")], 3)
        self.assertEqual(fixed_rooms[("Light", "Monday")], 2)


if __name__ == "__main__":
    unittest.main()
