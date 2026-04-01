"""Visualization and KPI utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_loader import (
    BLOCKED_PERIODS,
    DAY_NAMES,
    OPERATING_HOURS,
    ROOMS,
    SLOT_MINUTES,
    build_historical_schedule,
)
from .model import Solution

LOGGER = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def _minute_label(minute_value: int) -> str:
    """Format minute-of-day as HH:MM."""
    return f"{minute_value // 60:02d}:{minute_value % 60:02d}"


def _blocked_periods_for_day(day_name: str) -> list[tuple[int, int]]:
    """Return blocked periods for a named day."""
    return BLOCKED_PERIODS[DAY_NAMES.index(day_name)]


def _coerce_schedule_df(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize schedule fields used by plotting helpers."""
    if schedule_df.empty:
        return schedule_df.copy()
    normalized = schedule_df.copy()
    if "occupied_end_min" not in normalized.columns and "end_min" in normalized.columns:
        normalized["occupied_end_min"] = normalized["end_min"]
    if "room" in normalized.columns:
        normalized["room"] = pd.to_numeric(normalized["room"], errors="coerce")
    return normalized


def plot_gantt_by_provider(schedule_df: pd.DataFrame, day: str, title: str, output_path: str) -> None:
    """Plot a provider-oriented Gantt chart for a single day."""
    day_df = _coerce_schedule_df(schedule_df)
    day_df = day_df[day_df["day"] == day].copy()
    providers = sorted(day_df["provider"].unique().tolist())
    if not providers:
        return
    room_palette = sns.color_palette("tab20", 16)
    open_min, close_min = OPERATING_HOURS[DAY_NAMES.index(day)]
    fig, ax = plt.subplots(figsize=(20, len(providers) * 0.5 + 2))
    provider_positions = {provider: idx for idx, provider in enumerate(providers)}

    for _, row in day_df.iterrows():
        y = provider_positions[row["provider"]]
        room_value = row.get("room")
        color = room_palette[int(room_value) - 1] if pd.notna(room_value) and int(room_value) in ROOMS else "#9AA0A6"
        width = row["end_min"] - row["start_min"]
        ax.barh(y=y, width=width, left=row["start_min"], height=0.7, color=color, edgecolor="black")
        label = f"{str(row['patient_id'])[:10]} ({width}m)"
        ax.text(row["start_min"] + width / 2, y, label, ha="center", va="center", fontsize=7, clip_on=True)

    for start, end in _blocked_periods_for_day(day):
        if end <= open_min or start >= close_min:
            continue
        ax.axvspan(max(start, open_min), min(end, close_min), facecolor="lightgrey", alpha=0.4, hatch="//")

    ax.set_yticks(list(provider_positions.values()))
    ax.set_yticklabels(providers)
    ax.set_xlim(open_min, close_min)
    ax.set_xticks(list(range(open_min, close_min + 1, 30)))
    ax.set_xticklabels([_minute_label(value) for value in range(open_min, close_min + 1, 30)], rotation=45)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Provider")
    handles = [
        plt.Line2D([0], [0], color=room_palette[idx], lw=8, label=f"ER{idx + 1}")
        for idx in range(16)
    ]
    handles.append(plt.Line2D([0], [0], color="#9AA0A6", lw=8, label="Unassigned"))
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", title="Rooms")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gantt_by_room(schedule_df: pd.DataFrame, day: str, title: str, output_path: str) -> None:
    """Plot a room-oriented Gantt chart, highlighting overlaps in red."""
    day_df = _coerce_schedule_df(schedule_df)
    day_df = day_df[(day_df["day"] == day) & day_df["room"].isin(ROOMS)].copy()
    if day_df.empty:
        return
    providers = sorted(day_df["provider"].unique().tolist())
    provider_colors = dict(zip(providers, sns.color_palette("husl", len(providers)), strict=False))
    open_min, close_min = OPERATING_HOURS[DAY_NAMES.index(day)]
    fig, ax = plt.subplots(figsize=(20, len(ROOMS) * 0.5 + 2))

    for room_idx, room in enumerate(ROOMS):
        room_df = day_df[day_df["room"] == room].sort_values("start_min")
        for _, row in room_df.iterrows():
            overlaps = room_df[
                (room_df["patient_id"] != row["patient_id"])
                & (room_df["start_min"] < row["end_min"])
                & (room_df["end_min"] > row["start_min"])
            ]
            color = "red" if not overlaps.empty else provider_colors[row["provider"]]
            width = row["end_min"] - row["start_min"]
            ax.barh(y=room_idx, width=width, left=row["start_min"], height=0.7, color=color, edgecolor="black")
            ax.text(row["start_min"] + width / 2, room_idx, str(row["provider"]), ha="center", va="center", fontsize=7, clip_on=True)

    for start, end in _blocked_periods_for_day(day):
        if end <= open_min or start >= close_min:
            continue
        ax.axvspan(max(start, open_min), min(end, close_min), facecolor="lightgrey", alpha=0.4, hatch="//")

    ax.set_yticks(range(len(ROOMS)))
    ax.set_yticklabels([f"ER{room}" for room in ROOMS])
    ax.set_xlim(open_min, close_min)
    ax.set_xticks(list(range(open_min, close_min + 1, 30)))
    ax.set_xticklabels([_minute_label(value) for value in range(open_min, close_min + 1, 30)], rotation=45)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Room")
    handles = [plt.Line2D([0], [0], color=color, lw=8, label=provider) for provider, color in provider_colors.items()]
    if handles:
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", title="Provider")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_room_utilization_heatmap(schedule_df: pd.DataFrame, output_path: str) -> None:
    """Plot room utilization heatmaps faceted by day."""
    schedule_df = _coerce_schedule_df(schedule_df)
    schedule_df = schedule_df[schedule_df["room"].isin(ROOMS)].copy()
    if schedule_df.empty:
        return
    unique_days = [day for day in DAY_NAMES if day in schedule_df["day"].unique()]
    fig, axes = plt.subplots(len(unique_days), 1, figsize=(22, max(4, 3 * len(unique_days))), squeeze=False)
    for axis, day in zip(axes.flatten(), unique_days, strict=False):
        day_df = schedule_df[schedule_df["day"] == day]
        open_min, close_min = OPERATING_HOURS[DAY_NAMES.index(day)]
        slot_labels = list(range(open_min, close_min, SLOT_MINUTES))
        heatmap = pd.DataFrame(0, index=[f"ER{room}" for room in ROOMS], columns=slot_labels)
        for _, row in day_df.iterrows():
            for minute_value in range(int(row["start_min"]), int(row["occupied_end_min"]), SLOT_MINUTES):
                if minute_value in heatmap.columns:
                    heatmap.loc[f"ER{int(row['room'])}", minute_value] += 1
        sns.heatmap(
            heatmap,
            ax=axis,
            cmap=sns.color_palette(["white", "#4C78A8", "#E45756"], as_cmap=True),
            cbar=True,
            vmin=0,
            vmax=max(2, int(heatmap.to_numpy().max())),
        )
        axis.set_title(f"Room Utilization Heatmap - {day}")
        tick_positions = np.arange(0, len(slot_labels), 6)
        axis.set_xticks(tick_positions + 0.5)
        axis.set_xticklabels([_minute_label(slot_labels[pos]) for pos in tick_positions], rotation=45, ha="right")
        axis.set_xlabel("Time")
        axis.set_ylabel("Room")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_historical_baseline(
    appointments_df: pd.DataFrame,
    room_assignments: dict[str, dict[str, dict[str, Any]]],
    output_dir: str,
    title_prefix: str = "Historical Baseline",
) -> pd.DataFrame:
    """Render baseline charts directly from the historical appointment data."""
    schedule_df = build_historical_schedule(appointments_df, room_assignments)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    schedule_df.to_csv(output_path / "historical_schedule.csv", index=False)

    for day_name in DAY_NAMES:
        if day_name not in schedule_df.get("day", pd.Series(dtype=str)).unique():
            continue
        plot_gantt_by_provider(
            schedule_df,
            day_name,
            f"{title_prefix} - {day_name} by Provider",
            str(output_path / f"historical_gantt_by_provider_{day_name.lower()}.png"),
        )
        plot_gantt_by_room(
            schedule_df,
            day_name,
            f"{title_prefix} - {day_name} by Room",
            str(output_path / f"historical_gantt_by_room_{day_name.lower()}.png"),
        )
    plot_room_utilization_heatmap(schedule_df, str(output_path / "historical_room_utilization_heatmap.png"))
    return schedule_df


def compute_kpis(solution: Solution, appointments_df: pd.DataFrame) -> dict[str, Any]:
    """Compute fixed-time room-assignment KPIs for a solution."""
    schedule_df = solution.schedule_df.copy()
    total_non_cancelled = len(appointments_df)
    scheduled_real = schedule_df[~schedule_df.get("is_phantom", False)].copy() if not schedule_df.empty else schedule_df
    coverage_rate = 0.0 if total_non_cancelled == 0 else len(scheduled_real) / total_non_cancelled

    if schedule_df.empty:
        return {
            "coverage_rate": coverage_rate,
            "rooms_used": 0,
            "avg_rooms_used_per_day": 0.0,
            "avg_provider_day_length": 0.0,
            "total_travel_distance": 0.0,
            "avg_room_utilization": 0.0,
            "idle_time_minutes": 0.0,
            "room_conflict_count": 0,
            "room_switches_per_provider_day": 0.0,
            "no_show_adjusted_utilization": coverage_rate * (1 - float(appointments_df["no_show"].mean() or 0.0)),
        }

    provider_day_groups = schedule_df.groupby(["provider", "date"], sort=False)
    valid_room_df = schedule_df[schedule_df["room"].isin(ROOMS)] if "room" in schedule_df.columns else schedule_df.iloc[0:0]
    rooms_used = int(valid_room_df["room"].nunique()) if not valid_room_df.empty else 0
    rooms_per_day = valid_room_df.groupby("date")["room"].nunique() if not valid_room_df.empty else pd.Series(dtype=float)
    avg_rooms_used_per_day = float(rooms_per_day.mean()) if not rooms_per_day.empty else 0.0
    spans: list[float] = []
    idle_times: list[float] = []
    travel_total = 0.0
    room_switch_counts: list[float] = []
    for _, group in provider_day_groups:
        ordered = group.sort_values("start_min")
        spans.append(float(ordered["end_min"].max() - ordered["start_min"].min()))
        idle = 0.0
        switches = 0.0
        for (_, previous), (_, current) in zip(ordered.iloc[:-1].iterrows(), ordered.iloc[1:].iterrows(), strict=False):
            idle += max(0.0, float(current["start_min"] - previous["end_min"]))
            if previous.get("room") != current.get("room"):
                switches += 1.0
        idle_times.append(idle)
        room_switch_counts.append(switches)
    avg_provider_day_length = float(np.mean(spans)) if spans else 0.0
    idle_time_minutes = float(np.sum(idle_times)) if idle_times else 0.0
    room_switches_per_provider_day = float(np.mean(room_switch_counts)) if room_switch_counts else 0.0

    if "room" in schedule_df.columns:
        for _, group in valid_room_df.groupby(["provider", "date"], sort=False):
            ordered = group.sort_values("start_min")
            for (_, previous), (_, current) in zip(ordered.iloc[:-1].iterrows(), ordered.iloc[1:].iterrows(), strict=False):
                if previous["room"] != current["room"]:
                    travel_total += abs(float(previous["room"]) - float(current["room"]))

    capacity_slots = 0
    for date_value in appointments_df["date"].dt.normalize().unique():
        day_idx = pd.Timestamp(date_value).dayofweek
        open_min, close_min = OPERATING_HOURS[day_idx]
        blocked = _blocked_periods_for_day(DAY_NAMES[day_idx])
        available_minutes = close_min - open_min - sum(max(0, min(end, close_min) - max(start, open_min)) for start, end in blocked)
        capacity_slots += (available_minutes // SLOT_MINUTES) * len(ROOMS)
    occupied_slots = 0
    for _, row in schedule_df[schedule_df["room"].isin(ROOMS)].iterrows():
        occupied_slots += int((row["occupied_end_min"] - row["start_min"]) / SLOT_MINUTES)
    avg_room_utilization = 0.0 if capacity_slots == 0 else occupied_slots / capacity_slots

    slot_usage = (
        schedule_df[schedule_df["room"].isin(ROOMS)]
        .assign(slot=lambda df: df.apply(lambda row: list(range(int(row["start_min"]), int(row["occupied_end_min"]), SLOT_MINUTES)), axis=1))
        .explode("slot")
        .groupby(["date", "room", "slot"])
        .size()
    )
    room_conflict_count = int((slot_usage > 1).sum()) if not slot_usage.empty else 0

    no_show_rate = float(appointments_df["no_show"].mean() or 0.0)
    kpis = {
        "coverage_rate": float(coverage_rate),
        "rooms_used": rooms_used,
        "avg_rooms_used_per_day": avg_rooms_used_per_day,
        "avg_provider_day_length": avg_provider_day_length,
        "total_travel_distance": float(travel_total),
        "avg_room_utilization": float(avg_room_utilization),
        "idle_time_minutes": idle_time_minutes,
        "room_conflict_count": room_conflict_count,
        "room_switches_per_provider_day": room_switches_per_provider_day,
        "no_show_adjusted_utilization": float(coverage_rate * (1 - no_show_rate)),
    }
    kpis.update(solution.kpis)
    return kpis


def compare_policies_table(results: dict[str, Solution], output_path: str) -> pd.DataFrame:
    """Save a policy comparison table as CSV and styled HTML."""
    rows = {policy_name: solution.kpis for policy_name, solution in results.items()}
    table = pd.DataFrame.from_dict(rows, orient="index")
    csv_path = Path(output_path)
    html_path = csv_path.with_suffix(".html")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path)

    maximize_columns = {
        "coverage_rate",
        "avg_room_utilization",
        "no_show_adjusted_utilization",
        "expected_utilization",
        "robustness",
    }

    def _highlight(series: pd.Series) -> list[str]:
        best = series.max() if series.name in maximize_columns else series.min()
        return ["font-weight: bold" if value == best else "" for value in series]

    styled = table.style.format(precision=4).apply(_highlight, axis=0)
    html_path.write_text(styled.to_html(), encoding="utf-8")
    return table


def plot_kpi_radar(results: dict[str, Solution], output_path: str) -> None:
    """Plot a radar chart of normalized KPI values."""
    if not results:
        return
    kpi_map = {
        "coverage": "coverage_rate",
        "room_efficiency": "room_switches_per_provider_day",
        "travel_efficiency": "total_travel_distance",
        "utilization": "avg_room_utilization",
        "robustness": "robustness",
    }
    data = pd.DataFrame({name: solution.kpis for name, solution in results.items()}).T
    for column in kpi_map.values():
        if column not in data.columns:
            data[column] = 0.0
    normalized = pd.DataFrame(index=data.index)
    for label, column in kpi_map.items():
        series = data[column].astype(float)
        if column in {"total_travel_distance", "room_switches_per_provider_day"}:
            series = series.max() - series
        spread = series.max() - series.min()
        normalized[label] = 1.0 if spread == 0 else (series - series.min()) / spread

    labels = list(normalized.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for policy_name, row in normalized.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=policy_name)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Policy KPI Radar")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
