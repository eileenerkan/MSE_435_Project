"""Scheduling policy definitions."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

from .data_loader import DAY_NAMES, ROOMS, get_blocked_periods

LOGGER = logging.getLogger(__name__)


def policy_a_single_room(
    room_assignments: dict[str, dict[str, dict[str, Any]]],
    appointments_df: pd.DataFrame,
    dist_matrix: np.ndarray,
) -> dict[str, Any]:
    """
    Lock each provider to a single room for the entire day.

    Providers sharing the same preferred room on a day are reassigned greedily
    to the nearest still-available room, prioritizing busier providers.
    """
    appointment_counts = (
        appointments_df.assign(day_name=appointments_df["date"].dt.day_name())
        .groupby(["provider", "day_name"])
        .size()
        .to_dict()
    )
    fixed_rooms: dict[tuple[str, str], int | None] = {}
    for day_name in DAY_NAMES:
        day_providers: list[tuple[int, str, int | None]] = []
        for provider, day_map in room_assignments.items():
            assignment = day_map.get(day_name)
            if assignment is None:
                continue
            preferred_room = assignment.get("am_room") or assignment.get("pm_room")
            volume = int(appointment_counts.get((provider, day_name), 0))
            day_providers.append((volume, provider, preferred_room))

        taken_rooms: set[int] = set()
        for volume, provider, preferred_room in sorted(day_providers, key=lambda item: (-item[0], item[1])):
            chosen_room = preferred_room
            if chosen_room in taken_rooms or chosen_room is None:
                candidate_rooms = ROOMS.copy()
                if preferred_room is not None:
                    candidate_rooms = sorted(
                        ROOMS,
                        key=lambda room_id: (float(dist_matrix[preferred_room - 1, room_id - 1]), room_id),
                    )
                available_candidates = [room_id for room_id in candidate_rooms if room_id not in taken_rooms]
                chosen_room = available_candidates[0] if available_candidates else preferred_room
            fixed_rooms[(provider, day_name)] = chosen_room
            if chosen_room is not None:
                taken_rooms.add(chosen_room)
    return {"name": "Policy A", "fixed_room_all_day": fixed_rooms}


def policy_b_cluster_rooms(
    dist_matrix: np.ndarray,
    proximity_threshold: float = 3.0,
) -> dict[str, Any]:
    """
    Allow rooms within the proximity threshold of the provider's primary room.
    """
    return {
        "name": "Policy B",
        "cluster_threshold": proximity_threshold,
        "dist_matrix": dist_matrix,
    }


def policy_c_blocked_days(blocked_schedule: dict[str, list[str]]) -> dict[str, Any]:
    """
    Block provider-day pairs from scheduling.
    """
    normalized = {
        provider: sorted({day for day in days if day in DAY_NAMES})
        for provider, days in blocked_schedule.items()
    }
    return {"name": "Policy C", "blocked_schedule": normalized}


def policy_d_admin_buffer(use_admin_for_appts: bool = True) -> dict[str, Any]:
    """
    Allow appointments already booked in admin windows to keep their fixed times.
    """
    return {
        "name": "Policy D",
        "allow_admin_overflow": use_admin_for_appts,
        "respect_blocked_periods": True,
    }


def policy_e_overbooking(
    appointments_df: pd.DataFrame,
    no_show_rate: float | None = None,
    buffer_factor: float = 1.15,
) -> dict[str, Any]:
    """
    Configure overbooking and expected utilization reporting.
    """
    observed_rate = float(no_show_rate) if no_show_rate is not None else float(appointments_df["no_show"].mean())
    observed_rate = max(0.0, min(observed_rate, 0.95))
    overbook_multiplier = max(buffer_factor, 1.0 / max(1e-6, 1 - observed_rate))
    phantom_counts: dict[tuple[str, str], int] = {}
    group_sizes = appointments_df.groupby(["provider", "date_str"]).size()
    for key, actual_count in group_sizes.items():
        target = int(round(actual_count * overbook_multiplier))
        phantom_counts[key] = max(0, target - int(actual_count))
    return {
        "name": "Policy E",
        "overbooking": True,
        "no_show_rate": observed_rate,
        "buffer_factor": buffer_factor,
        "overbook_multiplier": overbook_multiplier,
        "phantom_counts": phantom_counts,
    }


def policy_f_uncertainty_buffer(
    buffer_slots: int = 1,
    use_stochastic: bool = True,
) -> dict[str, Any]:
    """
    Add deterministic or stochastic duration buffers.
    """
    return {
        "name": "Policy F",
        "buffer_slots": buffer_slots,
        "use_stochastic_buffer": use_stochastic,
        "robustness_trials": 1000,
        "random_seed": 42,
    }


def build_docx_blocked_schedule(
    room_assignments: dict[str, dict[str, dict[str, Any]]],
    week_number: int,
) -> dict[str, list[str]]:
    """
    Create blocked-day schedule from DOCX availability data and week closures.
    """
    blocked: dict[str, list[str]] = defaultdict(list)
    for provider, assignments in room_assignments.items():
        for day_name, assignment in assignments.items():
            if not assignment.get("available", False):
                blocked[provider].append(day_name)
    if week_number == 1:
        for provider in room_assignments:
            if "Tuesday" not in blocked[provider]:
                blocked[provider].append("Tuesday")
    return dict(blocked)


def validate_blocked_day_appointments(
    appointments_df: pd.DataFrame,
    blocked_schedule: dict[str, list[str]],
) -> list[str]:
    """
    Return warnings for appointments found on blocked provider-days.
    """
    warnings: list[str] = []
    for provider, blocked_days in blocked_schedule.items():
        if not blocked_days:
            continue
        matches = appointments_df[
            (appointments_df["provider"] == provider)
            & (appointments_df["date"].dt.day_name().isin(blocked_days))
        ]
        if not matches.empty:
            warnings.append(
                f"{provider} has {len(matches)} appointments on blocked days: {sorted(set(blocked_days))}"
            )
    return warnings


def compute_overbooking_metrics(
    scheduled_count: int,
    total_capacity: int,
    no_show_rate: float,
) -> dict[str, float]:
    """
    Compute expected utilization and overload probability under no-show uncertainty.
    """
    if total_capacity <= 0:
        return {"expected_utilization": 0.0, "probability_of_overload": 0.0}
    expected_utilization = scheduled_count * (1 - no_show_rate) / total_capacity
    probability_of_overload = 1 - binom.cdf(total_capacity, scheduled_count, 1 - no_show_rate)
    return {
        "expected_utilization": float(expected_utilization),
        "probability_of_overload": float(probability_of_overload),
    }


def filter_appointments_for_policy(
    appointments_df: pd.DataFrame,
    policy_params: dict[str, Any],
) -> pd.DataFrame:
    """
    Filter fixed-time appointments that a policy disallows at blocked periods.

    In the fixed-time model, Policy D means admin-window appointments are allowed
    to remain at their historical times. Policies that explicitly respect blocked
    periods without admin overflow drop those rows before room assignment.
    """
    if not policy_params.get("respect_blocked_periods") or policy_params.get("allow_admin_overflow"):
        return appointments_df.copy()

    keep_mask: list[bool] = []
    for _, row in appointments_df.iterrows():
        start_min = int(row["start_min"])
        blocked_periods = get_blocked_periods(int(row["day_of_week"]), policy_params)
        keep_mask.append(all(not (start <= start_min < end) for start, end in blocked_periods))
    return appointments_df.loc[keep_mask].copy().reset_index(drop=True)
