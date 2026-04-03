"""Main CLI for clinic room-assignment optimization with fixed appointment times."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tabulate import tabulate

from .data_loader import DAY_NAMES, build_distance_matrix, load_appointments, load_room_assignments
from .model import ColumnGenerator, Solution
from .policies import (
    build_docx_blocked_schedule,
    compute_admin_buffer_analysis,
    compute_overbooking_metrics,
    policy_a_single_room,
    policy_b_cluster_rooms,
    policy_c_blocked_days,
    policy_d_admin_buffer,
    policy_e_overbooking,
    policy_f_uncertainty_buffer,
    filter_appointments_for_policy,
    validate_blocked_day_appointments,
)
from .visualize import (
    compare_policies_table,
    compute_kpis,
    plot_gantt_by_provider,
    plot_gantt_by_room,
    plot_historical_baseline,
    plot_kpi_radar,
    plot_room_utilization_heatmap,
)

LOGGER = logging.getLogger(__name__)

HISTORICAL_BASELINE = "Historical Baseline"
OPTIMAL_POLICY = "Optimal"

# Additional provider-day blocks applied on top of DOCX availability for Policy C (Week 1).
# Format: {provider: [day_name, ...]}
WEEK1_EXTRA_BLOCKS: dict[str, list[str]] = {
    "HPW114": ["Monday", "Thursday"],
    "HPW101": ["Wednesday"],
}


def configure_logging() -> None:
    """Configure timestamped logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_week_inputs(
    week_number: int,
    appointment_csv: str,
    assignment_docx: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, Any]]]]:
    """Load and pre-filter week-specific inputs."""
    appointments_df = load_appointments(appointment_csv)
    room_assignments = load_room_assignments(assignment_docx)

    appointments_df = appointments_df[appointments_df["appt_type"].str.upper() != "ADMIN TIME"].copy()
    if week_number == 1:
        appointments_df = appointments_df[appointments_df["date"].dt.day_name() != "Tuesday"].copy()
    appointments_df = appointments_df.reset_index(drop=True)
    return appointments_df, room_assignments


def enrich_solution(
    solution: Solution,
    appointments_df: pd.DataFrame,
    policy_params: dict[str, Any],
    dist_matrix: np.ndarray,
) -> Solution:
    """Compute KPI overlays for a solution."""
    extra_kpis: dict[str, Any] = {}
    extra_kpis["total_travel_distance"] = compute_travel_distance(solution.schedule_df, dist_matrix)
    if policy_params.get("overbooking"):
        total_capacity = len(solution.schedule_df)
        overbook_metrics = compute_overbooking_metrics(
            scheduled_count=len(solution.schedule_df),
            total_capacity=total_capacity,
            no_show_rate=float(policy_params.get("no_show_rate", 0.0)),
        )
        extra_kpis.update(overbook_metrics)
    if "buffer_slots" in policy_params:
        robustness = estimate_robustness(solution.schedule_df, appointments_df, int(policy_params.get("robustness_trials", 1000)))
        extra_kpis["robustness"] = robustness
    if policy_params.get("name") == "Policy D":
        buffer_analysis = compute_admin_buffer_analysis(solution.schedule_df)
        extra_kpis.update(buffer_analysis)
        print("\nPolicy D — Admin Buffer Analysis:")
        print(f"  At-risk appointments (end within 15 min of admin boundary): {buffer_analysis['at_risk_appointments']}")
        print(f"  Absorbed by admin buffer:   {buffer_analysis['absorbed_by_buffer']}")
        print(f"  Unabsorbed conflicts:        {buffer_analysis['unabsorbed_conflicts']}")
        print(f"  Buffer absorption rate:      {buffer_analysis['buffer_absorption_rate']:.1%}\n")
    solution.kpis.update(extra_kpis)
    solution.kpis = compute_kpis(solution, appointments_df)
    return solution


def estimate_robustness(schedule_df: pd.DataFrame, appointments_df: pd.DataFrame, trials: int) -> float:
    """Estimate proportion of provider-days with zero cascading delays."""
    if schedule_df.empty:
        return 0.0
    np.random.seed(42)
    merged = schedule_df.merge(
        appointments_df[["appointment_id", "duration"]],
        on="appointment_id",
        how="left",
        suffixes=("", "_orig"),
    )
    provider_days = list(merged.groupby(["provider", "date"], sort=False))
    if not provider_days:
        return 0.0
    successful_days = 0
    total_days = 0
    for _, group in provider_days:
        group = group.sort_values("start_min")
        total_days += trials
        planned_starts = group["start_min"].to_numpy(dtype=float)
        planned_ends = group["end_min"].to_numpy(dtype=float)
        base_durations = group["duration"].fillna((group["end_min"] - group["start_min"])).to_numpy(dtype=float)
        for _ in range(trials):
            current_time = planned_starts[0]
            delayed = False
            for idx, planned_start in enumerate(planned_starts):
                actual_start = max(current_time, planned_start)
                sigma = max(1.0, 0.2 * base_durations[idx])
                actual_duration = max(5.0, np.random.normal(base_durations[idx], sigma))
                current_time = actual_start + actual_duration
                if actual_start - planned_start > 0.0:
                    delayed = True
                    break
                if idx < len(planned_ends) - 1 and current_time > planned_starts[idx + 1]:
                    delayed = True
                    break
            if not delayed:
                successful_days += 1
    return successful_days / total_days if total_days else 0.0


def compute_travel_distance(schedule_df: pd.DataFrame, dist_matrix: np.ndarray) -> float:
    """Compute total provider travel distance across room changes."""
    if schedule_df.empty:
        return 0.0
    total = 0.0
    for _, group in schedule_df.groupby(["provider", "date"], sort=False):
        ordered = group.sort_values("start_min")
        previous_room = None
        for _, row in ordered.iterrows():
            room = int(row["room"])
            if previous_room is not None and previous_room != room:
                total += float(dist_matrix[previous_room - 1, room - 1])
            previous_room = room
    return total


def render_policy_outputs(
    solution: Solution,
    output_dir: Path,
    policy_name: str,
) -> None:
    """Write charts and schedule CSV for one policy result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    solution.schedule_df.to_csv(output_dir / "schedule.csv", index=False)
    for day_name in DAY_NAMES:
        if day_name not in solution.schedule_df.get("day", pd.Series(dtype=str)).unique():
            continue
        plot_gantt_by_provider(
            solution.schedule_df,
            day_name,
            f"{policy_name} - {day_name} by Provider",
            str(output_dir / f"gantt_by_provider_{day_name.lower()}.png"),
        )
        plot_gantt_by_room(
            solution.schedule_df,
            day_name,
            f"{policy_name} - {day_name} by Room",
            str(output_dir / f"gantt_by_room_{day_name.lower()}.png"),
        )
    plot_room_utilization_heatmap(solution.schedule_df, str(output_dir / "room_utilization_heatmap.png"))


def render_historical_outputs(
    appointments_df: pd.DataFrame,
    room_assignments: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> Solution:
    """Render the historical clinic schedule and compute its KPIs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_df = plot_historical_baseline(appointments_df, room_assignments, str(output_dir), title_prefix=HISTORICAL_BASELINE)
    solution = Solution(
        columns=[],
        obj_val=0.0,
        schedule_df=schedule_df,
        unassigned=[],
        kpis={},
        warnings=[],
    )
    solution.kpis = compute_kpis(solution, appointments_df)
    return solution


def _drop_blocked_day_appointments(
    appointments_df: pd.DataFrame,
    policy_params: dict[str, Any],
) -> pd.DataFrame:
    """Remove appointments on provider-days blocked by the policy and report drops."""
    blocked_schedule = policy_params.get("blocked_schedule", {})
    if not blocked_schedule:
        return appointments_df.copy()
    day_name_col = appointments_df["date"].dt.day_name()
    drop_mask = appointments_df.apply(
        lambda row: day_name_col[row.name] in blocked_schedule.get(row["provider"], []),
        axis=1,
    )
    dropped = appointments_df[drop_mask]
    if not dropped.empty:
        print("\nPolicy C — appointments dropped due to blocked provider-days:")
        for (provider, day), grp in dropped.groupby([dropped["provider"], day_name_col[drop_mask]]):
            print(f"  {provider} on {day}: {len(grp)} appointments dropped")
        print(f"  Total dropped: {len(dropped)} / {len(appointments_df)}\n")
    return appointments_df[~drop_mask].copy().reset_index(drop=True)


def run_single_policy(
    appointments_df: pd.DataFrame,
    room_assignments: dict[str, dict[str, dict[str, Any]]],
    dist_matrix: np.ndarray,
    policy_name: str,
    policy_params: dict[str, Any],
) -> Solution:
    """Solve one fixed-time room-assignment policy scenario."""
    LOGGER.info("Running %s", policy_name)
    policy_appointments_df = _drop_blocked_day_appointments(appointments_df, policy_params)
    policy_appointments_df = filter_appointments_for_policy(policy_appointments_df, policy_params)
    generator = ColumnGenerator(
        appointments_df=policy_appointments_df,
        room_assignments=room_assignments,
        dist_matrix=dist_matrix,
        policy_params=policy_params,
    )
    solution = generator.solve()
    solution = enrich_solution(solution, appointments_df, policy_params, dist_matrix)
    return solution


def available_policies(
    appointments_df: pd.DataFrame,
    room_assignments: dict[str, dict[str, dict[str, Any]]],
    dist_matrix: np.ndarray,
    week_number: int,
    policy_b_threshold: float = 3.0,
) -> dict[str, dict[str, Any]]:
    """Build all named policy parameter dictionaries."""
    blocked_schedule = build_docx_blocked_schedule(room_assignments, week_number)
    warnings = validate_blocked_day_appointments(appointments_df, blocked_schedule)
    for warning in warnings:
        LOGGER.warning(warning)

    admin_overflow_minutes = set(
        appointments_df[
            (
                ((appointments_df["date"].dt.dayofweek < 4) & (appointments_df["start_min"].between(540, 565)))
                | ((appointments_df["date"].dt.dayofweek == 4) & (appointments_df["start_min"].between(480, 505)))
            )
        ]["start_min"].tolist()
    )

    if week_number == 1:
        for provider, days in WEEK1_EXTRA_BLOCKS.items():
            existing = blocked_schedule.get(provider, [])
            blocked_schedule[provider] = sorted(set(existing) | set(days))

    return {
        OPTIMAL_POLICY: {},
        "Policy A": policy_a_single_room(room_assignments, appointments_df, dist_matrix),
        "Policy B": policy_b_cluster_rooms(dist_matrix, proximity_threshold=policy_b_threshold),
        "Policy C": policy_c_blocked_days(blocked_schedule),
        "Policy D": policy_d_admin_buffer(),
        "Policy E": policy_e_overbooking(appointments_df),
        "Policy F": policy_f_uncertainty_buffer(),
    }


def resolve_requested_policy(requested_policy: str, policy_map: dict[str, dict[str, Any]]) -> str:
    """Resolve CLI policy input against supported canonical names."""
    normalized = requested_policy.strip().lower()
    if normalized == "all":
        return "all"

    alias_map = {
        "historical": HISTORICAL_BASELINE,
        "historical baseline": HISTORICAL_BASELINE,
        "baseline historical": HISTORICAL_BASELINE,
        "optimal": OPTIMAL_POLICY,
        "unconstrained": OPTIMAL_POLICY,
        "baseline": OPTIMAL_POLICY,
    }
    for policy_name in policy_map:
        alias_map[policy_name.lower()] = policy_name
    return alias_map.get(normalized, requested_policy)


def run_week(
    week_number: int,
    appointments_df: pd.DataFrame,
    room_assignments: dict[str, dict[str, dict[str, Any]]],
    dist_matrix: np.ndarray,
    output_root: Path,
    requested_policy: str,
    policy_b_threshold: float = 3.0,
) -> dict[str, Solution]:
    """Run all requested policy scenarios for a week."""
    policy_map = available_policies(appointments_df, room_assignments, dist_matrix, week_number, policy_b_threshold=policy_b_threshold)
    resolved_policy = resolve_requested_policy(requested_policy, policy_map)

    results: dict[str, Solution] = {}
    if resolved_policy in {"all", HISTORICAL_BASELINE}:
        historical_solution = render_historical_outputs(
            appointments_df=appointments_df,
            room_assignments=room_assignments,
            output_dir=output_root / "historical_baseline",
        )
        results[HISTORICAL_BASELINE] = historical_solution

    if resolved_policy == "all":
        selected_policy_map = policy_map
    elif resolved_policy == HISTORICAL_BASELINE:
        selected_policy_map = {}
    else:
        if resolved_policy not in policy_map:
            available = ", ".join([HISTORICAL_BASELINE, *policy_map.keys()])
            raise KeyError(f"Unknown policy '{requested_policy}'. Available options: {available}, all")
        selected_policy_map = {resolved_policy: policy_map[resolved_policy]}

    for policy_name, params in selected_policy_map.items():
        solution = run_single_policy(appointments_df, room_assignments, dist_matrix, policy_name, params)
        results[policy_name] = solution
        render_policy_outputs(solution, output_root / policy_name.lower().replace(" ", "_"), policy_name)

    table = compare_policies_table(results, str(output_root / "policy_comparison_table.csv"))
    plot_kpi_radar(results, str(output_root / "kpi_radar_chart.png"))
    print(f"\nWeek {week_number} KPI Comparison")
    print(tabulate(table.reset_index().rename(columns={"index": "Policy"}), headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Clinic examination room scheduling optimization.")
    parser.add_argument("--week", choices=["1", "2", "all"], default="all")
    parser.add_argument("--policy", default="all", help="Policy name to run or 'all'.")
    parser.add_argument("--output", default="./results/")
    parser.add_argument("--week1-csv", default="/Users/eileenerkan/Desktop/435_Project/AppointmentDataWeek1.csv")
    parser.add_argument("--week2-csv", default="/Users/eileenerkan/Desktop/435_Project/AppointmentDataWeek2.csv")
    parser.add_argument("--week1-docx", default="/Users/eileenerkan/Desktop/435_Project/ProviderRoomAssignmentWeek1.docx")
    parser.add_argument("--week2-docx", default="/Users/eileenerkan/Desktop/435_Project/ProviderRoomAssignmentWeek2.docx")
    parser.add_argument("--threshold", type=float, default=3.0, help="Proximity threshold for Policy B cluster rooms (default: 3.0).")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    configure_logging()
    args = parse_args()
    dist_matrix = build_distance_matrix()

    week_inputs = {
        1: load_week_inputs(1, args.week1_csv, args.week1_docx),
        2: load_week_inputs(2, args.week2_csv, args.week2_docx),
    }

    weeks_to_run = [1, 2] if args.week == "all" else [int(args.week)]
    for week_number in weeks_to_run:
        appointments_df, room_assignments = week_inputs[week_number]
        output_root = Path(args.output) / f"week{week_number}"
        run_week(
            week_number=week_number,
            appointments_df=appointments_df,
            room_assignments=room_assignments,
            dist_matrix=dist_matrix,
            output_root=output_root,
            requested_policy=args.policy,
            policy_b_threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
