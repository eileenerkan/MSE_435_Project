from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from clinic_scheduler.model import Solution
from clinic_scheduler.scheduler import HISTORICAL_BASELINE, OPTIMAL_POLICY, resolve_requested_policy
from clinic_scheduler.visualize import compare_policies_table


class SchedulerTests(unittest.TestCase):
    def test_resolve_requested_policy_accepts_aliases(self) -> None:
        policy_map = {
            OPTIMAL_POLICY: {},
            "Policy A": {},
            "Policy B": {},
        }

        self.assertEqual(resolve_requested_policy("all", policy_map), "all")
        self.assertEqual(resolve_requested_policy("historical baseline", policy_map), HISTORICAL_BASELINE)
        self.assertEqual(resolve_requested_policy("historical", policy_map), HISTORICAL_BASELINE)
        self.assertEqual(resolve_requested_policy("optimal", policy_map), OPTIMAL_POLICY)
        self.assertEqual(resolve_requested_policy("unconstrained", policy_map), OPTIMAL_POLICY)
        self.assertEqual(resolve_requested_policy("baseline", policy_map), OPTIMAL_POLICY)
        self.assertEqual(resolve_requested_policy("policy a", policy_map), "Policy A")

    def test_compare_policies_table_preserves_report_order(self) -> None:
        results = {
            HISTORICAL_BASELINE: Solution(columns=[], obj_val=0.0, schedule_df=pd.DataFrame(), unassigned=[], kpis={"coverage_rate": 0.7}),
            OPTIMAL_POLICY: Solution(columns=[], obj_val=0.0, schedule_df=pd.DataFrame(), unassigned=[], kpis={"coverage_rate": 0.9}),
            "Policy A": Solution(columns=[], obj_val=0.0, schedule_df=pd.DataFrame(), unassigned=[], kpis={"coverage_rate": 0.85}),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            table = compare_policies_table(results, str(Path(tmp_dir) / "policy_comparison_table.csv"))

        self.assertEqual(table.index.tolist(), [HISTORICAL_BASELINE, OPTIMAL_POLICY, "Policy A"])


if __name__ == "__main__":
    unittest.main()
