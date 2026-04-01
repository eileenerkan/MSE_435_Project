from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from clinic_scheduler.data_loader import ProviderDayKey
from clinic_scheduler.model import (
    ColumnGenerator,
    FINAL_ILP_TIMEOUT_SECONDS,
    UNCOVERED_APPOINTMENT_PENALTY,
)


class ModelTests(unittest.TestCase):
    def test_schedule_feasibility_metrics_detect_conflicts_and_partial_coverage(self) -> None:
        appointments_df = pd.DataFrame(
            [
                {
                    "appointment_id": "a1",
                    "provider": "Dr A",
                    "date_str": "2026-03-30",
                    "date": pd.Timestamp("2026-03-30"),
                    "day_of_week": 0,
                    "patient_id": "P1",
                    "start_min": 570,
                    "duration": 20,
                    "slots": 4,
                },
                {
                    "appointment_id": "a2",
                    "provider": "Dr B",
                    "date_str": "2026-03-30",
                    "date": pd.Timestamp("2026-03-30"),
                    "day_of_week": 0,
                    "patient_id": "P2",
                    "start_min": 570,
                    "duration": 20,
                    "slots": 4,
                },
            ]
        )
        generator = ColumnGenerator(
            appointments_df=appointments_df,
            room_assignments={},
            dist_matrix=np.zeros((16, 16)),
            policy_params={},
        )
        schedule_df = pd.DataFrame(
            [
                {
                    "appointment_id": "a1",
                    "date": "2026-03-30",
                    "room": 1,
                    "start_min": 570,
                    "occupied_end_min": 590,
                },
                {
                    "appointment_id": "a1",
                    "date": "2026-03-30",
                    "room": 1,
                    "start_min": 575,
                    "occupied_end_min": 595,
                },
            ]
        )

        coverage_rate, room_conflict_count = generator._schedule_feasibility_metrics(schedule_df)

        self.assertEqual(coverage_rate, 0.5)
        self.assertGreater(room_conflict_count, 0)

    def test_final_ilp_timeout_constant_is_extended(self) -> None:
        self.assertEqual(FINAL_ILP_TIMEOUT_SECONDS, 600)

    def test_build_initial_columns_generates_multiple_variants(self) -> None:
        appointments_df = pd.DataFrame(
            [
                {
                    "appointment_id": "a1",
                    "provider": "Dr A",
                    "date_str": "2026-03-30",
                    "date": pd.Timestamp("2026-03-30"),
                    "day_of_week": 0,
                    "patient_id": "P1",
                    "start_min": 600,
                    "duration": 20,
                    "slots": 4,
                }
            ]
        )
        generator = ColumnGenerator(
            appointments_df=appointments_df,
            room_assignments={"Dr A": {"Monday": {"am_room": 1, "pm_room": 1, "available": True, "status": "FULL_DAY"}}},
            dist_matrix=np.zeros((16, 16)),
            policy_params={},
        )

        generator.build_initial_columns()
        key = ProviderDayKey(provider="Dr A", day="Monday", date_str="2026-03-30")

        self.assertGreaterEqual(len(generator.columns_by_key[key]), 20)

    def test_unconstrained_policy_allows_all_rooms(self) -> None:
        appointments_df = pd.DataFrame(
            [
                {
                    "appointment_id": "a1",
                    "provider": "Dr A",
                    "date_str": "2026-03-30",
                    "date": pd.Timestamp("2026-03-30"),
                    "day_of_week": 0,
                    "patient_id": "P1",
                    "start_min": 600,
                    "duration": 20,
                    "slots": 4,
                }
            ]
        )
        generator = ColumnGenerator(
            appointments_df=appointments_df,
            room_assignments={"Dr A": {"Monday": {"am_room": 7, "pm_room": 7, "available": True, "status": "FULL_DAY"}}},
            dist_matrix=np.zeros((16, 16)),
            policy_params={},
        )

        self.assertEqual(generator._allowed_rooms("Dr A", "Monday", 600), list(range(1, 17)))

    def test_non_room_policy_params_still_fall_back_to_all_rooms(self) -> None:
        appointments_df = pd.DataFrame(
            [
                {
                    "appointment_id": "a1",
                    "provider": "Dr A",
                    "date_str": "2026-03-30",
                    "date": pd.Timestamp("2026-03-30"),
                    "day_of_week": 0,
                    "patient_id": "P1",
                    "start_min": 600,
                    "duration": 20,
                    "slots": 4,
                }
            ]
        )
        generator = ColumnGenerator(
            appointments_df=appointments_df,
            room_assignments={"Dr A": {"Monday": {"am_room": 7, "pm_room": 7, "available": True, "status": "FULL_DAY"}}},
            dist_matrix=np.zeros((16, 16)),
            policy_params={"respect_blocked_periods": True},
        )

        self.assertEqual(generator._allowed_rooms("Dr A", "Monday", 600), list(range(1, 17)))

    def test_room_ordering_varies_across_variants(self) -> None:
        appointments_df = pd.DataFrame(
            [
                {
                    "appointment_id": "a1",
                    "provider": "Dr A",
                    "date_str": "2026-03-30",
                    "date": pd.Timestamp("2026-03-30"),
                    "day_of_week": 0,
                    "patient_id": "P1",
                    "start_min": 600,
                    "duration": 20,
                    "slots": 4,
                }
            ]
        )
        generator = ColumnGenerator(
            appointments_df=appointments_df,
            room_assignments={"Dr A": {"Monday": {"am_room": 7, "pm_room": 7, "available": True, "status": "FULL_DAY"}}},
            dist_matrix=np.arange(16 * 16, dtype=float).reshape(16, 16),
            policy_params={},
        )

        appointment = appointments_df.iloc[0].to_dict()
        ordered_assigned = generator._ordered_candidate_rooms("Dr A", "Monday", appointment, "assigned")
        ordered_nearest = generator._ordered_candidate_rooms("Dr A", "Monday", appointment, "nearest_3")

        self.assertEqual(sorted(ordered_assigned), list(range(1, 17)))
        self.assertEqual(sorted(ordered_nearest), list(range(1, 17)))
        self.assertEqual(ordered_assigned[0], 1)
        self.assertEqual(ordered_nearest[:3], [1, 2, 3])

    def test_build_column_assigns_conflicting_fixed_time_to_next_room(self) -> None:
        appointments_df = pd.DataFrame(
            [
                {
                    "appointment_id": "a1",
                    "provider": "Dr A",
                    "date_str": "2026-03-30",
                    "date": pd.Timestamp("2026-03-30"),
                    "day_of_week": 0,
                    "patient_id": "P1",
                    "start_min": 600,
                    "duration": 20,
                    "slots": 4,
                },
                {
                    "appointment_id": "a2",
                    "provider": "Dr A",
                    "date_str": "2026-03-30",
                    "date": pd.Timestamp("2026-03-30"),
                    "day_of_week": 0,
                    "patient_id": "P2",
                    "start_min": 600,
                    "duration": 20,
                    "slots": 4,
                },
            ]
        )
        generator = ColumnGenerator(
            appointments_df=appointments_df,
            room_assignments={"Dr A": {"Monday": {"am_room": 1, "pm_room": 1, "available": True, "status": "FULL_DAY"}}},
            dist_matrix=np.zeros((16, 16)),
            policy_params={},
        )
        key = ProviderDayKey(provider="Dr A", day="Monday", date_str="2026-03-30")

        column = generator._build_column_from_strategy(
            key=key,
            appointments=appointments_df.sort_values(["start_min", "patient_id"]).to_dict("records"),
            variant="assigned",
            room_mode="assigned",
        )

        self.assertIsNotNone(column)
        assert column is not None
        self.assertEqual(len(column.schedule), 2)
        self.assertEqual([item["room"] for item in column.schedule], [1, 2])

    def test_fixed_room_policy_generates_single_room_column_only(self) -> None:
        appointments_df = pd.DataFrame(
            [
                {
                    "appointment_id": "a1",
                    "provider": "Dr A",
                    "date_str": "2026-03-30",
                    "date": pd.Timestamp("2026-03-30"),
                    "day_of_week": 0,
                    "patient_id": "P1",
                    "start_min": 600,
                    "duration": 20,
                    "slots": 4,
                },
                {
                    "appointment_id": "a2",
                    "provider": "Dr A",
                    "date_str": "2026-03-30",
                    "date": pd.Timestamp("2026-03-30"),
                    "day_of_week": 0,
                    "patient_id": "P2",
                    "start_min": 630,
                    "duration": 20,
                    "slots": 4,
                },
            ]
        )
        generator = ColumnGenerator(
            appointments_df=appointments_df,
            room_assignments={"Dr A": {"Monday": {"am_room": 5, "pm_room": 5, "available": True, "status": "FULL_DAY"}}},
            dist_matrix=np.zeros((16, 16)),
            policy_params={"fixed_room_all_day": {("Dr A", "Monday"): 5}},
        )
        key = ProviderDayKey(provider="Dr A", day="Monday", date_str="2026-03-30")

        generator.build_initial_columns()
        pricing_columns = generator.solve_pricing({})

        self.assertEqual(len(generator.columns_by_key[key]), 1)
        self.assertEqual(pricing_columns, [])
        assert generator.columns_by_key[key]
        self.assertEqual([item["room"] for item in generator.columns_by_key[key][0].schedule], [5, 5])

    def test_soft_coverage_master_problems_allow_uncovered_appointments(self) -> None:
        appointments_df = pd.DataFrame(
            [
                {
                    "appointment_id": "a1",
                    "provider": "Dr A",
                    "date_str": "2026-03-30",
                    "date": pd.Timestamp("2026-03-30"),
                    "day_of_week": 0,
                    "patient_id": "P1",
                    "start_min": 600,
                    "duration": 20,
                    "slots": 4,
                }
            ]
        )
        generator = ColumnGenerator(
            appointments_df=appointments_df,
            room_assignments={},
            dist_matrix=np.zeros((16, 16)),
            policy_params={},
        )

        rmp_obj, _ = generator.solve_rmp()
        _, ilp_obj, warning = generator._solve_final_ilp()

        self.assertEqual(rmp_obj, UNCOVERED_APPOINTMENT_PENALTY)
        self.assertEqual(ilp_obj, UNCOVERED_APPOINTMENT_PENALTY)
        self.assertIsNone(warning)


if __name__ == "__main__":
    unittest.main()
