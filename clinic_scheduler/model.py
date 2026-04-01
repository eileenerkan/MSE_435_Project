"""Column-generation-style scheduling model."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pulp

from .data_loader import DAY_NAMES, INDEX_TO_DAY, ProviderDayKey, ROOMS, SLOT_MINUTES, ceil_to_slot, floor_to_slot

LOGGER = logging.getLogger(__name__)

ALPHA = 0.0
BETA = 1.0
UNCOVERED_APPOINTMENT_PENALTY = 1000.0
FINAL_ILP_TIMEOUT_SECONDS = 600
INITIAL_ROOM_MODES = ["assigned", "nearest_3", "nearest_6", "all_rooms"]
PRICING_ROOM_MODES = ["room_assigned", "room_nearest", "room_cluster_2", "room_cluster_3", "room_cluster_4"]
MIN_INITIAL_COLUMNS_PER_PROVIDER_DAY = 4


def _safe_name(value: str) -> str:
    """Sanitize a string for PuLP constraint names."""
    return (
        value.replace("-", "_")
        .replace("|", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("/", "_")
    )


@dataclass
class Column:
    """A feasible provider-day schedule column."""

    provider: str
    day: str
    date_str: str
    appointments: list[str]
    schedule: list[dict[str, Any]]
    cost: float
    reduced_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def column_id(self) -> str:
        """Return a stable column identifier."""
        variant = self.metadata.get("variant", "base")
        return f"{self.provider}|{self.date_str}|{variant}|{len(self.schedule)}"


@dataclass
class Solution:
    """Final scheduling solution."""

    columns: list[Column]
    obj_val: float
    schedule_df: pd.DataFrame
    unassigned: list[str]
    kpis: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


class ColumnGenerator:
    """Restricted master problem plus heuristic pricing for room scheduling."""

    def __init__(
        self,
        appointments_df: pd.DataFrame,
        room_assignments: dict[str, dict[str, dict[str, Any]]],
        dist_matrix: np.ndarray,
        policy_params: dict[str, Any] | None = None,
    ) -> None:
        self.appointments_df = appointments_df.copy()
        self.room_assignments = room_assignments
        self.dist_matrix = dist_matrix
        self.policy_params = policy_params or {}
        self.columns_by_key: dict[ProviderDayKey, list[Column]] = defaultdict(list)
        self.all_columns: list[Column] = []
        self._column_lookup: dict[str, Column] = {}
        self._rmp_problem: pulp.LpProblem | None = None
        self._rmp_lambda: dict[str, pulp.LpVariable] = {}
        self._coverage_constraints: dict[str, pulp.LpConstraint] = {}
        self._provider_constraints: dict[ProviderDayKey, pulp.LpConstraint] = {}
        self._room_constraints: dict[tuple[str, int, int], pulp.LpConstraint] = {}
        self.warnings: list[str] = []
        self.phantom_counter = 0
        if self.policy_params.get("overbooking"):
            self._augment_with_phantoms()


    def _augment_with_phantoms(self) -> None:
        """Add phantom appointments for overbooking scenarios."""
        phantom_rows: list[dict[str, Any]] = []
        phantom_counts = self.policy_params.get("phantom_counts", {})
        base_groups = self.appointments_df.groupby(["provider", "date_str"], sort=False)
        for (provider, date_str), count in phantom_counts.items():
            if count <= 0:
                continue
            group = base_groups.get_group((provider, date_str)).sort_values("start_min")
            template_rows = group.to_dict("records")
            for index in range(count):
                template = dict(template_rows[index % len(template_rows)])
                self.phantom_counter += 1
                template["appointment_id"] = f"PHANTOM-APPT-{provider}-{date_str}-{self.phantom_counter}"
                template["patient_id"] = f"PHANTOM-{provider}-{date_str}-{self.phantom_counter}"
                template["status"] = "PHANTOM"
                template["no_show"] = False
                template["is_phantom"] = True
                phantom_rows.append(template)
        if phantom_rows:
            phantom_df = pd.DataFrame(phantom_rows)
            self.appointments_df["is_phantom"] = self.appointments_df.get("is_phantom", False)
            self.appointments_df = pd.concat([self.appointments_df, phantom_df], ignore_index=True)


    def build_initial_columns(self) -> None:
        """Create warm-start room-assignment columns for each provider-day."""
        grouped = self.appointments_df.groupby(["provider", "date_str"], sort=False)
        for (provider, date_str), group in grouped:
            day_index = int(group["day_of_week"].iloc[0])
            day_name = INDEX_TO_DAY[day_index]
            key = ProviderDayKey(provider=provider, day=day_name, date_str=date_str)
            if self._is_provider_day_blocked(provider, day_name):
                self.warnings.append(f"{provider} blocked on {day_name} {date_str}; appointments will remain unassigned.")
                continue
            appointments = group.sort_values(["start_min", "patient_id"]).to_dict("records")
            fixed_room = self._fixed_room_for_day(provider, day_name)
            room_modes = ["assigned"] if fixed_room is not None else INITIAL_ROOM_MODES
            for room_mode in room_modes:
                column = self._build_column_from_strategy(
                    key=key,
                    appointments=appointments,
                    variant="fixed_room" if fixed_room is not None else room_mode,
                    room_mode=room_mode,
                )
                if column is not None:
                    self._register_column(key, column)
            if fixed_room is None:
                for preferred_room in ROOMS:
                    column = self._build_column_from_strategy(
                        key=key,
                        appointments=appointments,
                        variant=f"anchor_{preferred_room}",
                        room_mode="all_rooms",
                        preferred_room=preferred_room,
                    )
                    if column is not None:
                        self._register_column(key, column)
            initial_count = len(self.columns_by_key[key])
            if initial_count < MIN_INITIAL_COLUMNS_PER_PROVIDER_DAY:
                warning = (
                    f"{provider} on {date_str} generated only {initial_count} initial columns; "
                    "the ILP may have too little schedule diversity for a feasible optimum."
                )
                LOGGER.warning(warning)
                self.warnings.append(warning)
        LOGGER.info(
            "Initial column pool built with %s columns across %s provider-days",
            len(self.all_columns),
            len(self.columns_by_key),
        )


    def _register_column(self, key: ProviderDayKey, column: Column) -> None:
        """Store a unique column."""
        if column.column_id in self._column_lookup:
            return
        self.columns_by_key[key].append(column)
        self.all_columns.append(column)
        self._column_lookup[column.column_id] = column


    def _is_provider_day_blocked(self, provider: str, day_name: str) -> bool:
        """Check blocked-day policy."""
        blocked_schedule = self.policy_params.get("blocked_schedule", {})
        return day_name in blocked_schedule.get(provider, [])


    def _get_assignment(self, provider: str, day_name: str) -> dict[str, Any]:
        """Return provider assignment metadata for a day."""
        if provider not in self.room_assignments:
            warning = f"Provider {provider} missing from room assignment table; using nearest available room fallback."
            if warning not in self.warnings:
                self.warnings.append(warning)
        return self.room_assignments.get(provider, {}).get(
            day_name,
            {"am_room": None, "pm_room": None, "available": True, "status": "MISSING"},
        )


    def _fixed_room_for_day(self, provider: str, day_name: str) -> int | None:
        """Return the fixed room required by Policy A for a provider-day."""
        fixed_room_map = self.policy_params.get("fixed_room_all_day")
        if fixed_room_map is None:
            return None
        return fixed_room_map.get((provider, day_name))


    def _primary_room(self, provider: str, day_name: str, appointment_start: int) -> int | None:
        """Return the primary assigned room for an appointment."""
        assignment = self._get_assignment(provider, day_name)
        if appointment_start < 720:
            return assignment.get("am_room") or assignment.get("pm_room")
        return assignment.get("pm_room") or assignment.get("am_room")


    def _allowed_rooms(self, provider: str, day_name: str, appointment_start: int) -> list[int]:
        """Return policy-compliant allowed rooms."""
        fixed_room = self._fixed_room_for_day(provider, day_name)
        if self.policy_params.get("fixed_room_all_day") is not None:
            if fixed_room is not None:
                return [fixed_room]
            central_room = min(
                ROOMS,
                key=lambda room_id: float(np.mean(self.dist_matrix[room_id - 1, :])),
            )
            return [central_room]

        if not self.policy_params or self.policy_params.get("allow_all_rooms", False):
            return ROOMS.copy()

        primary_room = self._primary_room(provider, day_name, appointment_start)
        if primary_room is None:
            return ROOMS.copy()

        if "cluster_threshold" in self.policy_params:
            threshold = float(self.policy_params["cluster_threshold"])
            clustered_rooms = [
                room
                for room in ROOMS
                if self.dist_matrix[primary_room - 1, room - 1] <= threshold
            ]
            return clustered_rooms or ROOMS.copy()

        return ROOMS.copy()


    def _buffered_slots(self, appointment: dict[str, Any]) -> int:
        """Return occupied slots including uncertainty buffers."""
        base_slots = int(appointment["slots"])
        if "buffer_slots" not in self.policy_params:
            return base_slots
        if self.policy_params.get("use_stochastic_buffer", False):
            sigma = max(1.0, 0.2 * float(appointment["duration"]))
            extra_slots = math.ceil(1.5 * sigma / SLOT_MINUTES)
        else:
            extra_slots = int(self.policy_params.get("buffer_slots", 0))
        return base_slots + max(0, extra_slots)


    def _fixed_start_min(self, appointment: dict[str, Any]) -> int:
        """Round a historical appointment start to the nearest slot boundary."""
        raw_start = int(appointment["start_min"])
        lower = floor_to_slot(raw_start)
        upper = ceil_to_slot(raw_start)
        return lower if raw_start - lower <= upper - raw_start else upper


    def _ordered_candidate_rooms(
        self,
        provider: str,
        day_name: str,
        appointment: dict[str, Any],
        room_mode: str,
        preferred_room: int | None = None,
    ) -> list[int]:
        """Return all 16 rooms ordered by the policy's room preference."""
        start_min = self._fixed_start_min(appointment)
        primary_room = self._primary_room(provider, day_name, start_min)
        policy_rooms = self._allowed_rooms(provider, day_name, start_min)
        if len(policy_rooms) == 1:
            return policy_rooms.copy()
        ordered_all_rooms = ROOMS.copy()
        if primary_room is not None:
            ordered_all_rooms = sorted(
                ROOMS,
                key=lambda room_id: (self.dist_matrix[primary_room - 1, room_id - 1], room_id),
            )

        preferred_rooms = [room for room in ordered_all_rooms if room in policy_rooms]
        fallback_rooms = [room for room in ordered_all_rooms if room not in policy_rooms]

        if room_mode in {"nearest_3", "room_cluster_2"}:
            preferred_rooms = preferred_rooms[:3] + [room for room in preferred_rooms[3:]]
        elif room_mode in {"nearest_6", "room_cluster_3"}:
            preferred_rooms = preferred_rooms[:6] + [room for room in preferred_rooms[6:]]
        elif room_mode == "room_cluster_4":
            preferred_rooms = preferred_rooms[:4] + [room for room in preferred_rooms[4:]]

        ordered_rooms = preferred_rooms + fallback_rooms
        if preferred_room is not None and preferred_room in ordered_rooms:
            ordered_rooms = [preferred_room] + [room for room in ordered_rooms if room != preferred_room]
        return ordered_rooms


    def _build_column_from_strategy(
        self,
        key: ProviderDayKey,
        appointments: list[dict[str, Any]],
        variant: str,
        room_mode: str,
        duals: dict[str, float] | None = None,
        preferred_room: int | None = None,
    ) -> Column | None:
        """Create one fixed-time room-assignment column."""
        schedule: list[dict[str, Any]] = []
        room_usage: set[tuple[int, int]] = set()
        prev_room: int | None = None
        for appointment in appointments:
            start_min = self._fixed_start_min(appointment)
            appt_slots = self._buffered_slots(appointment)
            occupied_slots = list(range(start_min, start_min + appt_slots * SLOT_MINUTES, SLOT_MINUTES))
            room_candidates = self._ordered_candidate_rooms(
                provider=key.provider,
                day_name=key.day,
                appointment=appointment,
                room_mode=room_mode,
                preferred_room=preferred_room,
            )
            best_choice: tuple[float, int, int] | None = None
            for room_rank, room in enumerate(room_candidates):
                has_conflict = any((room, slot) in room_usage for slot in occupied_slots)
                if has_conflict and len(room_candidates) > 1:
                    continue
                travel = 0.0 if prev_room is None else float(self.dist_matrix[prev_room - 1, room - 1])
                if duals is None:
                    score = BETA * travel
                else:
                    score = (
                        BETA * travel
                        - duals.get(f"cover::{appointment['appointment_id']}", 0.0)
                        - sum(duals.get(f"room::{key.date_str}::{room}::{slot}", 0.0) for slot in occupied_slots)
                    )
                if has_conflict:
                    warning = (
                        f"Fixed-room conflict for {appointment['appointment_id']} in room {room} "
                        f"at {start_min} on {key.date_str}; keeping the assignment."
                    )
                    LOGGER.warning(warning)
                    if warning not in self.warnings:
                        self.warnings.append(warning)
                candidate = (score, room_rank, room)
                best_choice = min(best_choice, candidate) if best_choice else candidate
            if best_choice is None:
                message = (
                    f"No conflict-free room exists for {appointment['appointment_id']} at fixed start "
                    f"{start_min} on {key.date_str}. All 16 rooms are occupied."
                )
                LOGGER.error(message)
                raise RuntimeError(message)
            _, _, chosen_room = best_choice
            buffered_end = start_min + appt_slots * SLOT_MINUTES
            real_end = start_min + int(appointment["slots"]) * SLOT_MINUTES
            for slot in occupied_slots:
                room_usage.add((chosen_room, slot))
            prev_room = chosen_room
            schedule.append(
                {
                    "appt_id": appointment["patient_id"],
                    "appointment_id": appointment["appointment_id"],
                    "patient_id": appointment["patient_id"],
                    "start_min": start_min,
                    "end_min": real_end,
                    "occupied_end_min": buffered_end,
                    "room": chosen_room,
                    "slots": int(appointment["slots"]),
                    "buffered_slots": appt_slots,
                    "duration": int(appointment["duration"]),
                    "original_start_min": int(appointment["start_min"]),
                    "is_phantom": bool(appointment.get("is_phantom", False)),
                }
            )
        if not schedule:
            return None
        cost = self._compute_column_cost(schedule)
        return Column(
            provider=key.provider,
            day=key.day,
            date_str=key.date_str,
            appointments=[item["appointment_id"] for item in schedule],
            schedule=schedule,
            cost=cost,
            metadata={"variant": variant},
        )


    def _compute_column_cost(self, schedule: list[dict[str, Any]]) -> float:
        """Compute room-switch travel cost for a fixed-time column."""
        sorted_schedule = sorted(schedule, key=lambda item: (item["start_min"], item["room"]))
        travel = 0.0
        for previous, current in zip(sorted_schedule, sorted_schedule[1:]):
            travel += float(self.dist_matrix[previous["room"] - 1, current["room"] - 1])
        return ALPHA * 0.0 + BETA * travel


    def _column_occupancy_keys(self, column: Column) -> set[tuple[str, int, int]]:
        """Return unique room-slot occupancies contributed by a column."""
        occupancy: set[tuple[str, int, int]] = set()
        for item in column.schedule:
            for slot in range(item["start_min"], item["occupied_end_min"], SLOT_MINUTES):
                occupancy.add((column.date_str, item["room"], slot))
        return occupancy


    def solve_rmp(self) -> tuple[float, dict[str, float]]:
        """Solve the restricted master LP and return objective plus dual values."""
        problem = pulp.LpProblem("ClinicSchedulingRMP", pulp.LpMinimize)
        lambda_vars = {
            column.column_id: pulp.LpVariable(f"lambda_{idx}", lowBound=0, upBound=1, cat="Continuous")
            for idx, column in enumerate(self.all_columns)
        }
        appointment_ids = self.appointments_df["appointment_id"].tolist()
        uncovered_vars = {
            appointment_id: pulp.LpVariable(f"u_{idx}", lowBound=0, upBound=1, cat="Continuous")
            for idx, appointment_id in enumerate(appointment_ids)
        }
        problem += (
            pulp.lpSum(column.cost * lambda_vars[column.column_id] for column in self.all_columns)
            + UNCOVERED_APPOINTMENT_PENALTY * pulp.lpSum(uncovered_vars.values())
        )

        coverage_constraints: dict[str, pulp.LpConstraint] = {}
        for appointment_id in appointment_ids:
            matching_columns = [
                lambda_vars[column.column_id]
                for column in self.all_columns
                if appointment_id in column.appointments
            ]
            constraint = pulp.lpSum(matching_columns) + uncovered_vars[appointment_id] == 1
            name = f"cover_{_safe_name(appointment_id)}"
            problem += constraint, name
            coverage_constraints[appointment_id] = problem.constraints[name]

        provider_constraints: dict[ProviderDayKey, pulp.LpConstraint] = {}
        for key, columns in self.columns_by_key.items():
            name = f"provider_{_safe_name(key.provider)}_{_safe_name(key.date_str)}_{_safe_name(key.day)}"
            constraint = pulp.lpSum(lambda_vars[column.column_id] for column in columns) <= 1
            problem += constraint, name
            provider_constraints[key] = problem.constraints[name]

        room_constraints: dict[tuple[str, int, int], pulp.LpConstraint] = {}
        occupancy_map: dict[tuple[str, int, int], list[pulp.LpVariable]] = defaultdict(list)
        for column in self.all_columns:
            for occupancy_key in self._column_occupancy_keys(column):
                occupancy_map[occupancy_key].append(lambda_vars[column.column_id])
        for (date_str, room, slot), variables in occupancy_map.items():
            name = f"room_{_safe_name(date_str)}_{room}_{slot}"
            constraint = pulp.lpSum(variables) <= 1
            problem += constraint, name
            room_constraints[(date_str, room, slot)] = problem.constraints[name]

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
        problem.solve(solver)
        duals: dict[str, float] = {}
        for appointment_id, constraint in coverage_constraints.items():
            duals[f"cover::{appointment_id}"] = float(constraint.pi or 0.0)
        for key, constraint in provider_constraints.items():
            duals[f"provider::{key.provider}::{key.date_str}"] = float(constraint.pi or 0.0)
        for room_key, constraint in room_constraints.items():
            duals[f"room::{room_key[0]}::{room_key[1]}::{room_key[2]}"] = float(constraint.pi or 0.0)

        self._rmp_problem = problem
        self._rmp_lambda = lambda_vars
        self._coverage_constraints = coverage_constraints
        self._provider_constraints = provider_constraints
        self._room_constraints = room_constraints
        return float(pulp.value(problem.objective)), duals


    def solve_pricing(self, duals: dict[str, float]) -> list[Column]:
        """Generate new room-assignment columns with negative reduced cost."""
        new_columns: list[Column] = []
        grouped = self.appointments_df.groupby(["provider", "date_str"], sort=False)
        for (provider, date_str), group in grouped:
            day_name = group["date"].dt.day_name().iloc[0]
            key = ProviderDayKey(provider=provider, day=day_name, date_str=date_str)
            if self._is_provider_day_blocked(provider, day_name):
                continue
            if self._fixed_room_for_day(provider, day_name) is not None:
                continue
            appointments = group.sort_values(["start_min", "patient_id"]).to_dict("records")
            for room_mode in PRICING_ROOM_MODES:
                column = self._build_column_from_strategy(
                    key=key,
                    appointments=appointments,
                    variant=f"pricing_{room_mode}",
                    room_mode=room_mode,
                    duals=duals,
                )
                if column is None:
                    continue
                reduced_cost = self._reduced_cost(column, duals)
                column.reduced_cost = reduced_cost
                if reduced_cost < -1e-6 and column.column_id not in self._column_lookup:
                    new_columns.append(column)
        return new_columns


    def _reduced_cost(self, column: Column, duals: dict[str, float]) -> float:
        """Compute reduced cost for a candidate column."""
        dual_sum = duals.get(f"provider::{column.provider}::{column.date_str}", 0.0)
        for appointment_id in column.appointments:
            dual_sum += duals.get(f"cover::{appointment_id}", 0.0)
        for date_str, room, slot in self._column_occupancy_keys(column):
            dual_sum += duals.get(f"room::{date_str}::{room}::{slot}", 0.0)
        return float(column.cost - dual_sum)


    def solve(self, max_iter: int = 50) -> Solution:
        """Run column generation and finalize with an ILP over generated columns."""
        self.build_initial_columns()
        if not self.all_columns:
            return Solution(columns=[], obj_val=0.0, schedule_df=pd.DataFrame(), unassigned=[], kpis={}, warnings=self.warnings)

        best_obj = math.inf
        for iteration in range(max_iter):
            obj_val, duals = self.solve_rmp()
            LOGGER.info("Column generation iteration %s objective %.3f columns=%s", iteration + 1, obj_val, len(self.all_columns))
            best_obj = min(best_obj, obj_val)
            new_columns = self.solve_pricing(duals)
            if not new_columns:
                break
            for column in new_columns:
                key = ProviderDayKey(provider=column.provider, day=column.day, date_str=column.date_str)
                self._register_column(key, column)

        selected_columns, ilp_obj, warning = self._solve_final_ilp()
        if warning:
            self.warnings.append(warning)
        schedule_df = self._build_schedule_df(selected_columns)
        appointment_ids = set(self.appointments_df["appointment_id"].tolist())
        covered = set(schedule_df["appointment_id"].tolist()) if not schedule_df.empty else set()
        unassigned = sorted(appointment_ids - covered)
        coverage_rate, room_conflict_count = self._schedule_feasibility_metrics(schedule_df)
        if room_conflict_count > 0 or coverage_rate < 1.0:
            warning = (
                "Final ILP returned a schedule with feasibility issues: "
                f"coverage_rate={coverage_rate:.4f}, room_conflict_count={room_conflict_count}."
            )
            LOGGER.warning(warning)
            self.warnings.append(warning)
        return Solution(
            columns=selected_columns,
            obj_val=ilp_obj,
            schedule_df=schedule_df,
            unassigned=unassigned,
            kpis={},
            warnings=self.warnings,
        )


    def _solve_final_ilp(self) -> tuple[list[Column], float, str | None]:
        """Solve integer master problem over final columns."""
        problem = pulp.LpProblem("ClinicSchedulingILP", pulp.LpMinimize)
        lambda_vars = {
            column.column_id: pulp.LpVariable(f"y_{idx}", lowBound=0, upBound=1, cat="Binary")
            for idx, column in enumerate(self.all_columns)
        }

        appointment_ids = self.appointments_df["appointment_id"].tolist()
        uncovered_vars = {
            appointment_id: pulp.LpVariable(f"u_{idx}", lowBound=0, upBound=1, cat="Binary")
            for idx, appointment_id in enumerate(appointment_ids)
        }
        problem += (
            pulp.lpSum(column.cost * lambda_vars[column.column_id] for column in self.all_columns)
            + UNCOVERED_APPOINTMENT_PENALTY * pulp.lpSum(uncovered_vars.values())
        )
        for appointment_id in appointment_ids:
            matching_columns = [
                lambda_vars[column.column_id]
                for column in self.all_columns
                if appointment_id in column.appointments
            ]
            problem += (
                pulp.lpSum(matching_columns) + uncovered_vars[appointment_id] == 1,
                f"cover_{_safe_name(appointment_id)}",
            )

        for key, columns in self.columns_by_key.items():
            problem += (
                pulp.lpSum(lambda_vars[column.column_id] for column in columns) <= 1,
                f"provider_{_safe_name(key.provider)}_{_safe_name(key.date_str)}_{_safe_name(key.day)}",
            )

        occupancy_map: dict[tuple[str, int, int], list[pulp.LpVariable]] = defaultdict(list)
        for column in self.all_columns:
            for occupancy_key in self._column_occupancy_keys(column):
                occupancy_map[occupancy_key].append(lambda_vars[column.column_id])
        for (date_str, room, slot), variables in occupancy_map.items():
            problem += pulp.lpSum(variables) <= 1, f"room_{_safe_name(date_str)}_{room}_{slot}"

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=FINAL_ILP_TIMEOUT_SECONDS)
        status = problem.solve(solver)
        status_name = pulp.LpStatus.get(status, "Unknown")
        warning = None
        if status_name not in {"Optimal", "Integer Feasible"}:
            warning = f"ILP finished with status {status_name}; returning best available solution."
            return [], float(pulp.value(problem.objective) or 0.0), warning
        selected = []
        for column_id, variable in lambda_vars.items():
            value = float(variable.value() or 0.0)
            if value > 0.5:
                selected.append(self._column_lookup[column_id])
        return selected, float(pulp.value(problem.objective) or 0.0), warning


    def _schedule_feasibility_metrics(self, schedule_df: pd.DataFrame) -> tuple[float, int]:
        """Return simple feasibility diagnostics for a finalized schedule."""
        total_appointments = int(self.appointments_df["appointment_id"].nunique())
        covered_appointments = int(schedule_df["appointment_id"].nunique()) if not schedule_df.empty else 0
        coverage_rate = 1.0 if total_appointments == 0 else covered_appointments / total_appointments

        if schedule_df.empty:
            return coverage_rate, 0

        slot_usage = (
            schedule_df.assign(
                slot=lambda df: df.apply(
                    lambda row: list(range(int(row["start_min"]), int(row["occupied_end_min"]), SLOT_MINUTES)),
                    axis=1,
                )
            )
            .explode("slot")
            .groupby(["date", "room", "slot"])
            .size()
        )
        room_conflict_count = int((slot_usage > 1).sum()) if not slot_usage.empty else 0
        return coverage_rate, room_conflict_count


    def _build_schedule_df(self, columns: list[Column]) -> pd.DataFrame:
        """Flatten selected columns into a schedule dataframe."""
        rows: list[dict[str, Any]] = []
        for column in columns:
            for item in column.schedule:
                rows.append(
                    {
                        "provider": column.provider,
                        "day": column.day,
                        "date": column.date_str,
                        "appointment_id": item["appointment_id"],
                        "patient_id": item["patient_id"],
                        "start_min": item["start_min"],
                        "end_min": item["end_min"],
                        "occupied_end_min": item["occupied_end_min"],
                        "room": item["room"],
                        "slots": item["slots"],
                        "buffered_slots": item["buffered_slots"],
                        "is_phantom": item["is_phantom"],
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "provider",
                    "day",
                    "date",
                    "appointment_id",
                    "patient_id",
                    "start_min",
                    "end_min",
                    "occupied_end_min",
                    "room",
                    "slots",
                    "buffered_slots",
                    "is_phantom",
                ]
            )
        return pd.DataFrame(rows).sort_values(["date", "provider", "start_min", "room"]).reset_index(drop=True)
