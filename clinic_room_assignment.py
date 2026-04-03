from itertools import combinations

import pandas as pd
import pulp


DEFAULT_TIME_LIMIT_SECONDS = 120


def appointment_intervals_overlap(start1, end1, start2, end2):
    """Return True when two half-open time intervals overlap."""
    return (start1 < end2) and (start2 < end1)


def load_and_prepare_appointments(csv_path, keep_no_shows=False):
    """
    Load the appointment file and apply baseline preprocessing.

    Rules:
    - drop cancelled appointments
    - drop deleted appointments
    - optionally drop no-shows
    - drop ADMIN TIME appointments
    - drop appointments on 2025-11-11 (Tuesday Week 1 — clinic closed)
    - build scheduled start/end timestamps from Appt Date + Appt Time + Appt Duration
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    if "Cancelled Appts" in df.columns:
        df = df[df["Cancelled Appts"] != "Y"].copy()

    if "Deleted Appts" in df.columns:
        df = df[df["Deleted Appts"] != "Y"].copy()

    if not keep_no_shows and "No Show Appts" in df.columns:
        df = df[df["No Show Appts"] != "Y"].copy()

    if "Appt Type" in df.columns:
        df = df[df["Appt Type"].str.upper() != "ADMIN TIME"].copy()

    df["Appt Date"] = pd.to_datetime(df["Appt Date"], format="%m-%d-%Y", errors="coerce")
    df = df[df["Appt Date"] != pd.Timestamp("2025-11-11")].copy()

    df["Appt Time"] = df["Appt Time"].astype(str).str.strip()
    df["Appt Start"] = pd.to_datetime(
        df["Appt Date"].dt.strftime("%Y-%m-%d") + " " + df["Appt Time"],
        errors="coerce",
    )
    df["Appt Duration"] = pd.to_numeric(df["Appt Duration"], errors="coerce")
    df = df.dropna(subset=["Appt Date", "Appt Start", "Appt Duration"]).copy()

    df["Appt End"] = df["Appt Start"] + pd.to_timedelta(df["Appt Duration"], unit="m")
    df["Weekday"] = df["Appt Date"].dt.day_name()
    df = df.reset_index(drop=True)
    df["appt_id"] = [f"A{i}" for i in range(len(df))]
    df["day_str"] = df["Appt Date"].dt.strftime("%Y-%m-%d")

    return df


def build_feasible_room_map(appts_df, candidate_rooms):
    """
    Baseline room eligibility map.

    Right now every appointment can use every candidate room.
    This function is kept separate so later policies can narrow room eligibility.
    """
    feasible_map = {}
    all_rooms = list(candidate_rooms)
    for appt_id in appts_df["appt_id"]:
        feasible_map[appt_id] = list(all_rooms)
    return feasible_map, all_rooms


def build_overlap_pairs(appts_df):
    """Build pairs of appointments that overlap in scheduled time on the same day."""
    overlap_pairs = []
    records = appts_df[
        ["appt_id", "Appt Date", "Appt Start", "Appt End"]
    ].to_dict(orient="records")

    for a, b in combinations(records, 2):
        if a["Appt Date"] != b["Appt Date"]:
            continue
        if appointment_intervals_overlap(
            a["Appt Start"], a["Appt End"], b["Appt Start"], b["Appt End"]
        ):
            overlap_pairs.append((a["appt_id"], b["appt_id"]))

    return overlap_pairs


def build_same_provider_overlap_pairs(appts_df):
    """Return overlapping appointment pairs for the same provider on the same day."""
    overlap_pairs = []
    cols = [
        "appt_id",
        "Primary Provider",
        "Appt Date",
        "Appt Start",
        "Appt End",
        "day_str",
    ]
    records = appts_df[cols].to_dict(orient="records")

    for a, b in combinations(records, 2):
        if a["Primary Provider"] != b["Primary Provider"]:
            continue
        if a["Appt Date"] != b["Appt Date"]:
            continue
        if appointment_intervals_overlap(
            a["Appt Start"], a["Appt End"], b["Appt Start"], b["Appt End"]
        ):
            overlap_pairs.append((a["appt_id"], b["appt_id"]))

    return overlap_pairs


def summarize_same_provider_overlaps(appts_df):
    """Summarize same-provider overlaps by provider and day."""
    overlap_rows = []
    lookup = appts_df.set_index("appt_id")[
        ["Primary Provider", "day_str", "Appt Start", "Appt End"]
    ]

    for i, j in build_same_provider_overlap_pairs(appts_df):
        row_i = lookup.loc[i]
        overlap_rows.append(
            {
                "appt_id_1": i,
                "appt_id_2": j,
                "Primary Provider": row_i["Primary Provider"],
                "Day": row_i["day_str"],
            }
        )

    if not overlap_rows:
        return pd.DataFrame(
            columns=["Primary Provider", "Day", "same_provider_overlap_pairs"]
        )

    detail_df = pd.DataFrame(overlap_rows)
    return (
        detail_df.groupby(["Primary Provider", "Day"], as_index=False)
        .size()
        .rename(columns={"size": "same_provider_overlap_pairs"})
        .sort_values(["same_provider_overlap_pairs", "Primary Provider", "Day"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def compute_peak_concurrency(appts_df):
    """Compute the peak number of simultaneous appointments by day."""
    results = []

    for day, group in appts_df.groupby("Appt Date"):
        events = []
        for _, row in group.iterrows():
            events.append((row["Appt Start"], 1))
            events.append((row["Appt End"], -1))

        # Process end events before start events at the same timestamp.
        events.sort(key=lambda x: (x[0], x[1]))

        running = 0
        peak = 0
        for _, delta in events:
            running += delta
            peak = max(peak, running)

        results.append({"Appt Date": day, "peak_concurrent_appointments": peak})

    return pd.DataFrame(results).sort_values("Appt Date").reset_index(drop=True)


def build_provider_day_room_summary(result_df):
    """Summarize how many appointments each provider-day uses in each assigned room."""
    if result_df.empty or "Assigned Room" not in result_df.columns:
        return pd.DataFrame(
            columns=["Primary Provider", "Day", "Assigned Room", "num_appointments"]
        )

    summary = (
        result_df.groupby(["Primary Provider", "day_str", "Assigned Room"], as_index=False)
        .size()
        .rename(columns={"day_str": "Day", "size": "num_appointments"})
        .sort_values(["Primary Provider", "Day", "Assigned Room"])
        .reset_index(drop=True)
    )
    return summary


def create_cbc_solver(msg=False, time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS):
    return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_seconds)


def validate_room_clusters(candidate_rooms, room_clusters):
    """
    Validate and normalize cluster definitions.

    Input format:
    {
        "Cluster A": ["Room 1", "Room 2", "Room 3"],
        "Cluster B": ["Room 4", "Room 5", "Room 6"],
    }
    """
    if not room_clusters:
        raise ValueError("room_clusters must be a non-empty dictionary.")

    candidate_room_set = set(candidate_rooms)
    normalized = {}
    seen_rooms = set()

    for cluster_name, rooms in room_clusters.items():
        room_list = list(rooms)
        if not room_list:
            raise ValueError(f"Cluster '{cluster_name}' is empty.")

        unknown_rooms = [room for room in room_list if room not in candidate_room_set]
        if unknown_rooms:
            raise ValueError(
                f"Cluster '{cluster_name}' contains rooms not in candidate_rooms: {unknown_rooms}"
            )

        overlap_rooms = [room for room in room_list if room in seen_rooms]
        if overlap_rooms:
            raise ValueError(
                f"Rooms cannot belong to multiple clusters. Overlap found in '{cluster_name}': {overlap_rooms}"
            )

        normalized[cluster_name] = room_list
        seen_rooms.update(room_list)

    uncovered_rooms = [room for room in candidate_rooms if room not in seen_rooms]
    if uncovered_rooms:
        raise ValueError(
            f"Every candidate room must belong to a cluster. Missing rooms: {uncovered_rooms}"
        )

    return normalized


def _add_baseline_constraints(model, appt_ids, all_rooms, feasible_map, overlap_pairs, x, y):
    # Each appointment must be assigned to exactly one feasible room.
    for appt_id in appt_ids:
        model += (
            pulp.lpSum(x[appt_id][room] for room in feasible_map[appt_id]) == 1,
            f"assign_{appt_id}",
        )

    # Two overlapping appointments cannot occupy the same room.
    for appt_i, appt_j in overlap_pairs:
        for room in all_rooms:
            model += (
                x[appt_i][room] + x[appt_j][room] <= 1,
                f"overlap_{appt_i}_{appt_j}_{room}",
            )

    # A room is marked active if any appointment is assigned to it.
    for appt_id in appt_ids:
        for room in feasible_map[appt_id]:
            model += (x[appt_id][room] <= y[room], f"activate_{appt_id}_{room}")


def _extract_assignment_df(appts_df, all_rooms, x_vars):
    assignments = []
    for appt_id in appts_df["appt_id"]:
        assigned_room = None
        for room in all_rooms:
            value = pulp.value(x_vars[appt_id][room])
            if value is not None and value > 0.5:
                assigned_room = room
                break
        assignments.append({"appt_id": appt_id, "Assigned Room": assigned_room})

    assign_df = pd.DataFrame(assignments)
    return appts_df.merge(assign_df, on="appt_id", how="left")


def _base_result_dict(
    scenario_name,
    policy_name,
    keep_no_shows,
    appts_df,
    overlap_pairs,
    same_provider_overlap_pairs,
):
    return {
        "scenario_name": scenario_name,
        "policy_name": policy_name,
        "keep_no_shows": keep_no_shows,
        "appointments_df": appts_df.copy(),
        "overlap_pairs": list(overlap_pairs),
        "same_provider_overlap_pairs": list(same_provider_overlap_pairs),
        "num_appointments": len(appts_df),
        "num_overlap_pairs": len(overlap_pairs),
        "num_same_provider_overlap_pairs": len(same_provider_overlap_pairs),
        "peak_concurrency_df": compute_peak_concurrency(appts_df),
        "same_provider_overlap_summary_df": summarize_same_provider_overlaps(appts_df),
    }


def solve_unrestricted_baseline(
    appts_df,
    candidate_rooms,
    keep_no_shows=False,
    solver_msg=False,
    time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
    scenario_name="Baseline - Minimize rooms used",
):
    """
    Baseline scenario: any appointment may use any candidate room.

    Math:
    - x[i, r] = 1 if appointment i uses room r
    - y[r] = 1 if room r is used
    - minimize sum_r y[r]
    """
    appts_df = appts_df.copy()
    feasible_map, all_rooms = build_feasible_room_map(appts_df, candidate_rooms)
    overlap_pairs = build_overlap_pairs(appts_df)
    same_provider_overlap_pairs = build_same_provider_overlap_pairs(appts_df)
    appt_ids = appts_df["appt_id"].tolist()

    model = pulp.LpProblem("Baseline_Room_Assignment_Unrestricted", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (appt_ids, all_rooms), cat="Binary")
    y = pulp.LpVariable.dicts("y", all_rooms, cat="Binary")

    model += pulp.lpSum(y[room] for room in all_rooms)
    _add_baseline_constraints(model, appt_ids, all_rooms, feasible_map, overlap_pairs, x, y)

    status_code = model.solve(create_cbc_solver(msg=solver_msg, time_limit_seconds=time_limit_seconds))
    status = pulp.LpStatus[status_code]

    result = _base_result_dict(
        scenario_name,
        "baseline_minimize_rooms",
        keep_no_shows,
        appts_df,
        overlap_pairs,
        same_provider_overlap_pairs,
    )
    result["solver_status"] = status
    result["model"] = model

    if status != "Optimal":
        result["feasible"] = False
        result["objective_value"] = None
        result["used_rooms"] = []
        result["assignments_df"] = pd.DataFrame()
        result["provider_day_room_usage_df"] = pd.DataFrame()
        result["notes"] = f"Baseline model ended with solver status '{status}'."
        return result

    used_rooms = [room for room in all_rooms if pulp.value(y[room]) > 0.5]
    assignments_df = _extract_assignment_df(appts_df, all_rooms, x)

    result["feasible"] = True
    result["objective_value"] = int(round(pulp.value(model.objective)))
    result["used_rooms"] = used_rooms
    result["assignments_df"] = assignments_df
    result["provider_day_room_usage_df"] = build_provider_day_room_summary(assignments_df)
    result["notes"] = (
        "Baseline solved successfully. This scenario only minimizes the number of rooms used."
    )
    return result


def solve_provider_room_cap_per_day(
    appts_df,
    candidate_rooms,
    max_rooms_per_provider_day,
    keep_no_shows=False,
    solver_msg=False,
    time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
    scenario_name=None,
):
    """
    Policy C: each provider may use at most k rooms per day.

    Math:
    - x[i, r] = 1 if appointment i uses room r
    - y[r] = 1 if room r is used anywhere
    - z[p, d, r] = 1 if provider p uses room r on day d
    - minimize sum_r y[r]
    - for each provider-day (p, d): sum_r z[p, d, r] <= k
    - for each appointment i of provider p on day d: x[i, r] <= z[p, d, r]
    """
    appts_df = appts_df.copy()
    feasible_map, all_rooms = build_feasible_room_map(appts_df, candidate_rooms)
    overlap_pairs = build_overlap_pairs(appts_df)
    same_provider_overlap_pairs = build_same_provider_overlap_pairs(appts_df)
    appt_ids = appts_df["appt_id"].tolist()

    provider_day_pairs = list(
        appts_df[["Primary Provider", "day_str"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    if scenario_name is None:
        scenario_name = f"Policy cap - Provider <= {max_rooms_per_provider_day} rooms/day"

    model = pulp.LpProblem(
        f"Provider_Room_Cap_{max_rooms_per_provider_day}_Per_Day",
        pulp.LpMinimize,
    )
    x = pulp.LpVariable.dicts("x", (appt_ids, all_rooms), cat="Binary")
    y = pulp.LpVariable.dicts("y", all_rooms, cat="Binary")
    z = pulp.LpVariable.dicts("z", (provider_day_pairs, all_rooms), cat="Binary")

    model += pulp.lpSum(y[room] for room in all_rooms)
    _add_baseline_constraints(model, appt_ids, all_rooms, feasible_map, overlap_pairs, x, y)

    for provider, day in provider_day_pairs:
        model += (
            pulp.lpSum(z[(provider, day)][room] for room in all_rooms)
            <= max_rooms_per_provider_day,
            f"provider_room_cap_{provider}_{day}",
        )

    for _, row in appts_df.iterrows():
        appt_id = row["appt_id"]
        provider = row["Primary Provider"]
        day = row["day_str"]
        for room in all_rooms:
            model += (
                x[appt_id][room] <= z[(provider, day)][room],
                f"provider_room_link_{appt_id}_{provider}_{day}_{room}",
            )

    status_code = model.solve(create_cbc_solver(msg=solver_msg, time_limit_seconds=time_limit_seconds))
    status = pulp.LpStatus[status_code]

    result = _base_result_dict(
        scenario_name,
        f"provider_at_most_{max_rooms_per_provider_day}_rooms_per_day",
        keep_no_shows,
        appts_df,
        overlap_pairs,
        same_provider_overlap_pairs,
    )
    result["solver_status"] = status
    result["model"] = model
    result["max_rooms_per_provider_day"] = max_rooms_per_provider_day

    if status != "Optimal":
        result["feasible"] = False
        result["objective_value"] = None
        result["used_rooms"] = []
        result["assignments_df"] = pd.DataFrame()
        result["provider_day_room_usage_df"] = pd.DataFrame()
        if max_rooms_per_provider_day == 1 and same_provider_overlap_pairs:
            result["notes"] = (
                "Infeasible. Same-provider overlapping appointments exist in the input, "
                "so some provider-days require more than one room under fixed appointment times."
            )
        else:
            result["notes"] = (
                f"Scenario ended with solver status '{status}'. "
                "Inspect provider-day overlaps and the room cap for diagnosis."
            )
        return result

    used_rooms = [room for room in all_rooms if pulp.value(y[room]) > 0.5]
    assignments_df = _extract_assignment_df(appts_df, all_rooms, x)

    result["feasible"] = True
    result["objective_value"] = int(round(pulp.value(model.objective)))
    result["used_rooms"] = used_rooms
    result["assignments_df"] = assignments_df
    result["provider_day_room_usage_df"] = build_provider_day_room_summary(assignments_df)
    result["notes"] = (
        f"Solved with provider room cap k={max_rooms_per_provider_day} per day."
    )
    return result


def solve_one_provider_one_room_per_day(
    appts_df,
    candidate_rooms,
    keep_no_shows=False,
    solver_msg=False,
    time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
    scenario_name="Policy A - One provider, one room/day",
):
    """
    Policy A: each provider may use at most one room per day.

    This is the validation policy that tests whether a provider can stay
    in the same room for the full day under fixed appointment times.
    """
    return solve_provider_room_cap_per_day(
        appts_df=appts_df,
        candidate_rooms=candidate_rooms,
        max_rooms_per_provider_day=1,
        keep_no_shows=keep_no_shows,
        solver_msg=solver_msg,
        time_limit_seconds=time_limit_seconds,
        scenario_name=scenario_name,
    )


def solve_provider_cluster_per_day(
    appts_df,
    candidate_rooms,
    room_clusters,
    keep_no_shows=False,
    solver_msg=False,
    time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
    scenario_name="Policy B - Provider assigned to one room cluster/day",
):
    """
    Policy B: each provider must stay within one proximity cluster of rooms per day.

    Math:
    - x[i, r] = 1 if appointment i uses room r
    - y[r] = 1 if room r is used anywhere
    - z[p, d, c] = 1 if provider p uses cluster c on day d
    - minimize sum_r y[r]
    - for each provider-day (p, d): sum_c z[p, d, c] = 1
    - for each appointment i of provider p on day d and room r in cluster c:
      x[i, r] <= z[p, d, c]
    """
    appts_df = appts_df.copy()
    feasible_map, all_rooms = build_feasible_room_map(appts_df, candidate_rooms)
    room_clusters = validate_room_clusters(all_rooms, room_clusters)
    overlap_pairs = build_overlap_pairs(appts_df)
    same_provider_overlap_pairs = build_same_provider_overlap_pairs(appts_df)
    appt_ids = appts_df["appt_id"].tolist()
    provider_day_pairs = list(
        appts_df[["Primary Provider", "day_str"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    cluster_names = list(room_clusters.keys())
    room_to_cluster = {
        room: cluster_name
        for cluster_name, rooms in room_clusters.items()
        for room in rooms
    }

    model = pulp.LpProblem("Provider_Cluster_Per_Day", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (appt_ids, all_rooms), cat="Binary")
    y = pulp.LpVariable.dicts("y", all_rooms, cat="Binary")
    z = pulp.LpVariable.dicts("z", (provider_day_pairs, cluster_names), cat="Binary")

    model += pulp.lpSum(y[room] for room in all_rooms)
    _add_baseline_constraints(model, appt_ids, all_rooms, feasible_map, overlap_pairs, x, y)

    for provider, day in provider_day_pairs:
        model += (
            pulp.lpSum(z[(provider, day)][cluster_name] for cluster_name in cluster_names) == 1,
            f"one_cluster_{provider}_{day}",
        )

    for _, row in appts_df.iterrows():
        appt_id = row["appt_id"]
        provider = row["Primary Provider"]
        day = row["day_str"]
        for room in all_rooms:
            cluster_name = room_to_cluster[room]
            model += (
                x[appt_id][room] <= z[(provider, day)][cluster_name],
                f"cluster_link_{appt_id}_{provider}_{day}_{room}",
            )

    status_code = model.solve(create_cbc_solver(msg=solver_msg, time_limit_seconds=time_limit_seconds))
    status = pulp.LpStatus[status_code]

    result = _base_result_dict(
        scenario_name,
        "provider_one_cluster_per_day",
        keep_no_shows,
        appts_df,
        overlap_pairs,
        same_provider_overlap_pairs,
    )
    result["solver_status"] = status
    result["model"] = model
    result["room_clusters"] = room_clusters

    if status != "Optimal":
        result["feasible"] = False
        result["objective_value"] = None
        result["used_rooms"] = []
        result["assignments_df"] = pd.DataFrame()
        result["provider_day_room_usage_df"] = pd.DataFrame()
        result["notes"] = (
            f"Cluster policy ended with solver status '{status}'. "
            "Try larger clusters or inspect provider-day overlap patterns."
        )
        return result

    used_rooms = [room for room in all_rooms if pulp.value(y[room]) > 0.5]
    assignments_df = _extract_assignment_df(appts_df, all_rooms, x)
    assignments_df["Assigned Cluster"] = assignments_df["Assigned Room"].map(room_to_cluster)

    cluster_assignments = []
    for provider, day in provider_day_pairs:
        assigned_cluster = None
        for cluster_name in cluster_names:
            value = pulp.value(z[(provider, day)][cluster_name])
            if value is not None and value > 0.5:
                assigned_cluster = cluster_name
                break
        cluster_assignments.append(
            {
                "Primary Provider": provider,
                "Day": day,
                "Assigned Cluster": assigned_cluster,
            }
        )

    result["feasible"] = True
    result["objective_value"] = int(round(pulp.value(model.objective)))
    result["used_rooms"] = used_rooms
    result["assignments_df"] = assignments_df
    result["provider_day_room_usage_df"] = build_provider_day_room_summary(assignments_df)
    result["provider_day_cluster_df"] = pd.DataFrame(cluster_assignments)
    result["notes"] = (
        "Solved with a one-cluster-per-provider-per-day policy."
    )
    return result


# ─── Admin window helpers ─────────────────────────────────────────────────────

def _admin_boundary_minutes(weekday_name):
    """
    Return admin window boundary points (minutes since midnight) for a given weekday.

    Mon–Thu: morning ends 9:30, noon admin 11:30–12:00, lunch 12:00–13:00, afternoon starts 16:30
    Fri:     morning ends 8:30, noon admin 11:30–12:00, lunch 12:00–13:00, afternoon starts 15:00
    """
    noon_boundaries = [11 * 60 + 30, 12 * 60, 13 * 60]
    if weekday_name == "Friday":
        return [8 * 60 + 30] + noon_boundaries + [15 * 60]
    return [9 * 60 + 30] + noon_boundaries + [16 * 60 + 30]


def _minutes_since_midnight(ts):
    return ts.hour * 60 + ts.minute + ts.second / 60.0


# ─── Task 4 Stage 2: Refinement model ─────────────────────────────────────────

def solve_refinement_model(
    appts_df,
    candidate_rooms,
    r_star,
    keep_no_shows=False,
    solver_msg=False,
    time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
    scenario_name="Stage 2 - Minimize provider-day-room assignments",
):
    """
    Stage 2: Fix total rooms to r_star; minimize provider-day-room fragmentation.

    Math:
    - x[i, r] = 1 if appointment i uses room r
    - y[r] = 1 if room r is used
    - z[p, d, r] = 1 if provider p uses room r on day d
    - sum_r y[r] == r_star
    - minimize sum_{p,d,r} z[p, d, r]
    - x[i, r] <= z[p, d, r] for appointment i of provider p on day d
    """
    appts_df = appts_df.copy()
    feasible_map, all_rooms = build_feasible_room_map(appts_df, candidate_rooms)
    overlap_pairs = build_overlap_pairs(appts_df)
    same_provider_overlap_pairs = build_same_provider_overlap_pairs(appts_df)
    appt_ids = appts_df["appt_id"].tolist()

    provider_day_pairs = list(
        appts_df[["Primary Provider", "day_str"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    model = pulp.LpProblem("Refinement_Minimize_Provider_Day_Room", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (appt_ids, all_rooms), cat="Binary")
    y = pulp.LpVariable.dicts("y", all_rooms, cat="Binary")
    z = pulp.LpVariable.dicts("z", (provider_day_pairs, all_rooms), cat="Binary")

    model += pulp.lpSum(z[(p, d)][room] for p, d in provider_day_pairs for room in all_rooms)
    _add_baseline_constraints(model, appt_ids, all_rooms, feasible_map, overlap_pairs, x, y)

    # Fix total rooms used to exactly r_star.
    model += (
        pulp.lpSum(y[room] for room in all_rooms) == r_star,
        "fix_total_rooms",
    )

    # Prevent y[r]=1 for rooms with no appointment assigned (needed since y is not minimized).
    for room in all_rooms:
        model += (
            y[room] <= pulp.lpSum(x[appt_id][room] for appt_id in appt_ids),
            f"room_active_only_if_used_{room}",
        )

    # Link appointment-room assignment to provider-day-room indicator.
    for _, row in appts_df.iterrows():
        appt_id = row["appt_id"]
        provider = row["Primary Provider"]
        day = row["day_str"]
        for room in all_rooms:
            model += (
                x[appt_id][room] <= z[(provider, day)][room],
                f"refine_link_{appt_id}_{provider}_{day}_{room}",
            )

    status_code = model.solve(create_cbc_solver(msg=solver_msg, time_limit_seconds=time_limit_seconds))
    status = pulp.LpStatus[status_code]

    result = _base_result_dict(
        scenario_name,
        "refinement_minimize_provider_day_room",
        keep_no_shows,
        appts_df,
        overlap_pairs,
        same_provider_overlap_pairs,
    )
    result["solver_status"] = status
    result["model"] = model
    result["r_star"] = r_star

    if status != "Optimal":
        result["feasible"] = False
        result["objective_value"] = None
        result["used_rooms"] = []
        result["assignments_df"] = pd.DataFrame()
        result["provider_day_room_usage_df"] = pd.DataFrame()
        result["notes"] = f"Refinement model ended with solver status '{status}'."
        return result

    used_rooms = [room for room in all_rooms if pulp.value(y[room]) > 0.5]
    assignments_df = _extract_assignment_df(appts_df, all_rooms, x)

    result["feasible"] = True
    result["objective_value"] = int(round(pulp.value(model.objective)))
    result["used_rooms"] = used_rooms
    result["assignments_df"] = assignments_df
    result["provider_day_room_usage_df"] = build_provider_day_room_summary(assignments_df)
    result["notes"] = (
        f"Stage 2 solved. Rooms fixed at r_star={r_star}. "
        "Minimized provider-day-room assignments."
    )
    return result


# ─── Task 4: Two-stage wrapper ─────────────────────────────────────────────────

def solve_two_stage(
    csv_path,
    candidate_rooms,
    keep_no_shows=False,
    solver_msg=False,
    time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
):
    """
    Two-stage solve:
    Stage 1 — unrestricted baseline to find r_star (minimum rooms needed).
    Stage 2 — fix rooms to r_star, minimize provider-day-room assignments.

    Returns (stage1_result, stage2_result).
    If stage 1 is infeasible, stage2_result is None.
    """
    appts_df = load_and_prepare_appointments(csv_path, keep_no_shows=keep_no_shows)

    stage1_result = solve_unrestricted_baseline(
        appts_df=appts_df,
        candidate_rooms=candidate_rooms,
        keep_no_shows=keep_no_shows,
        solver_msg=solver_msg,
        time_limit_seconds=time_limit_seconds,
    )

    if not stage1_result["feasible"]:
        return stage1_result, None

    r_star = stage1_result["objective_value"]
    stage2_result = solve_refinement_model(
        appts_df=appts_df,
        candidate_rooms=candidate_rooms,
        r_star=r_star,
        keep_no_shows=keep_no_shows,
        solver_msg=solver_msg,
        time_limit_seconds=time_limit_seconds,
    )

    return stage1_result, stage2_result


# ─── Task 4c: Blocked days ─────────────────────────────────────────────────────

def solve_blocked_days(
    appts_df,
    candidate_rooms,
    blocked_days,
    keep_no_shows=False,
    solver_msg=False,
    time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
    scenario_name="Blocked days - Provider availability constraints",
):
    """
    Task 4c: Drop appointments for blocked providers on their blocked days,
    then solve baseline room minimization on the remaining appointments.

    blocked_days: dict mapping provider name to list of date strings (MM-DD-YYYY).
    e.g. {"HPW114": ["11-10-2025", "11-13-2025"]}
    """
    appts_df = appts_df.copy()

    # Build a set of (provider, day_str) pairs to block.
    blocked_set = set()
    for provider, date_strs in blocked_days.items():
        for date_str in date_strs:
            dt = pd.to_datetime(date_str, format="%m-%d-%Y", errors="coerce")
            if pd.isna(dt):
                dt = pd.to_datetime(date_str, errors="coerce")
            if not pd.isna(dt):
                blocked_set.add((provider, dt.strftime("%Y-%m-%d")))

    mask = appts_df.apply(
        lambda row: (row["Primary Provider"], row["day_str"]) in blocked_set, axis=1
    )
    blocked_appts_df = appts_df[mask].copy()
    remaining_df = appts_df[~mask].copy().reset_index(drop=True)
    remaining_df["appt_id"] = [f"A{i}" for i in range(len(remaining_df))]

    feasible_map, all_rooms = build_feasible_room_map(remaining_df, candidate_rooms)
    overlap_pairs = build_overlap_pairs(remaining_df)
    same_provider_overlap_pairs = build_same_provider_overlap_pairs(remaining_df)
    appt_ids = remaining_df["appt_id"].tolist()

    model = pulp.LpProblem("Blocked_Days_Room_Assignment", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (appt_ids, all_rooms), cat="Binary")
    y = pulp.LpVariable.dicts("y", all_rooms, cat="Binary")

    model += pulp.lpSum(y[room] for room in all_rooms)
    _add_baseline_constraints(model, appt_ids, all_rooms, feasible_map, overlap_pairs, x, y)

    status_code = model.solve(create_cbc_solver(msg=solver_msg, time_limit_seconds=time_limit_seconds))
    status = pulp.LpStatus[status_code]

    result = _base_result_dict(
        scenario_name,
        "blocked_days_baseline",
        keep_no_shows,
        remaining_df,
        overlap_pairs,
        same_provider_overlap_pairs,
    )
    result["solver_status"] = status
    result["model"] = model
    result["blocked_days"] = blocked_days
    result["blocked_appointments_df"] = blocked_appts_df

    if status != "Optimal":
        result["feasible"] = False
        result["objective_value"] = None
        result["used_rooms"] = []
        result["assignments_df"] = pd.DataFrame()
        result["provider_day_room_usage_df"] = pd.DataFrame()
        result["notes"] = f"Blocked-days model ended with solver status '{status}'."
        return result

    used_rooms = [room for room in all_rooms if pulp.value(y[room]) > 0.5]
    assignments_df = _extract_assignment_df(remaining_df, all_rooms, x)

    result["feasible"] = True
    result["objective_value"] = int(round(pulp.value(model.objective)))
    result["used_rooms"] = used_rooms
    result["assignments_df"] = assignments_df
    result["provider_day_room_usage_df"] = build_provider_day_room_summary(assignments_df)
    result["notes"] = (
        f"Blocked-days solve complete. {len(blocked_appts_df)} appointment(s) dropped "
        f"for {len(blocked_days)} provider(s)."
    )
    return result


# ─── Task 4d: Admin buffer ─────────────────────────────────────────────────────

def solve_admin_buffer(
    appts_df,
    candidate_rooms,
    admin_buffer_minutes,
    keep_no_shows=False,
    solver_msg=False,
    time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
    scenario_name=None,
):
    """
    Task 4d: Solve the baseline model, then flag appointments whose start or
    end time falls within admin_buffer_minutes of any admin window boundary.

    Admin windows (hard-coded per spec):
    Mon–Thu: morning admin ends 9:30, noon admin 11:30–12:00,
             lunch 12:00–13:00, afternoon admin starts 16:30
    Fri:     morning admin ends 8:30, noon admin 11:30–12:00,
             lunch 12:00–13:00, afternoon admin starts 15:00
    """
    if scenario_name is None:
        scenario_name = f"Admin buffer - {admin_buffer_minutes} min around admin windows"

    appts_df = appts_df.copy()
    feasible_map, all_rooms = build_feasible_room_map(appts_df, candidate_rooms)
    overlap_pairs = build_overlap_pairs(appts_df)
    same_provider_overlap_pairs = build_same_provider_overlap_pairs(appts_df)
    appt_ids = appts_df["appt_id"].tolist()

    model = pulp.LpProblem("Admin_Buffer_Room_Assignment", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (appt_ids, all_rooms), cat="Binary")
    y = pulp.LpVariable.dicts("y", all_rooms, cat="Binary")

    model += pulp.lpSum(y[room] for room in all_rooms)
    _add_baseline_constraints(model, appt_ids, all_rooms, feasible_map, overlap_pairs, x, y)

    status_code = model.solve(create_cbc_solver(msg=solver_msg, time_limit_seconds=time_limit_seconds))
    status = pulp.LpStatus[status_code]

    result = _base_result_dict(
        scenario_name,
        "admin_buffer_baseline",
        keep_no_shows,
        appts_df,
        overlap_pairs,
        same_provider_overlap_pairs,
    )
    result["solver_status"] = status
    result["model"] = model
    result["admin_buffer_minutes"] = admin_buffer_minutes

    if status != "Optimal":
        result["feasible"] = False
        result["objective_value"] = None
        result["used_rooms"] = []
        result["assignments_df"] = pd.DataFrame()
        result["provider_day_room_usage_df"] = pd.DataFrame()
        result["num_near_admin_window"] = None
        result["notes"] = f"Admin buffer model ended with solver status '{status}'."
        return result

    used_rooms = [room for room in all_rooms if pulp.value(y[room]) > 0.5]
    assignments_df = _extract_assignment_df(appts_df, all_rooms, x)

    # Post-process: flag appointments near any admin window boundary.
    def _is_near_admin_window(row):
        boundaries = _admin_boundary_minutes(row["Weekday"])
        start_m = _minutes_since_midnight(row["Appt Start"])
        end_m = _minutes_since_midnight(row["Appt End"])
        return any(
            abs(start_m - b) <= admin_buffer_minutes or abs(end_m - b) <= admin_buffer_minutes
            for b in boundaries
        )

    assignments_df["near_admin_window"] = assignments_df.apply(_is_near_admin_window, axis=1)
    num_near = int(assignments_df["near_admin_window"].sum())

    result["feasible"] = True
    result["objective_value"] = int(round(pulp.value(model.objective)))
    result["used_rooms"] = used_rooms
    result["assignments_df"] = assignments_df
    result["provider_day_room_usage_df"] = build_provider_day_room_summary(assignments_df)
    result["num_near_admin_window"] = num_near
    result["notes"] = (
        f"Admin buffer analysis complete. {num_near} appointment(s) fall within "
        f"{admin_buffer_minutes} min of an admin window boundary."
    )
    return result


# ─── Task 4e: Overbooking ──────────────────────────────────────────────────────

def solve_overbooking(
    appts_df_with_noshows,
    appts_df_without_noshows,
    candidate_rooms,
    no_show_rate,
    solver_msg=False,
    time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
    scenario_name="Overbooking - scheduled vs realized rooms",
):
    """
    Task 4e: Quantify room savings from no-shows.

    Solve baseline on appts_df_with_noshows  → scheduled_rooms (worst-case planning).
    Solve baseline on appts_df_without_noshows → actual_rooms_used (realized demand).
    rooms_saved = scheduled_rooms - actual_rooms_used
    """
    scheduled_result = solve_unrestricted_baseline(
        appts_df=appts_df_with_noshows,
        candidate_rooms=candidate_rooms,
        keep_no_shows=True,
        solver_msg=solver_msg,
        time_limit_seconds=time_limit_seconds,
        scenario_name=f"{scenario_name} [scheduled]",
    )
    actual_result = solve_unrestricted_baseline(
        appts_df=appts_df_without_noshows,
        candidate_rooms=candidate_rooms,
        keep_no_shows=False,
        solver_msg=solver_msg,
        time_limit_seconds=time_limit_seconds,
        scenario_name=f"{scenario_name} [actual]",
    )

    scheduled_rooms = scheduled_result["objective_value"]
    actual_rooms_used = actual_result["objective_value"]
    num_no_shows = len(appts_df_with_noshows) - len(appts_df_without_noshows)
    rooms_saved = (
        (scheduled_rooms - actual_rooms_used)
        if scheduled_rooms is not None and actual_rooms_used is not None
        else None
    )

    result = scheduled_result.copy()
    result["scenario_name"] = scenario_name
    result["policy_name"] = "overbooking_analysis"
    result["no_show_rate"] = no_show_rate
    result["num_no_shows"] = num_no_shows
    result["scheduled_rooms"] = scheduled_rooms
    result["actual_rooms_used"] = actual_rooms_used
    result["rooms_saved"] = rooms_saved
    result["actual_result"] = actual_result
    result["notes"] = (
        f"Overbooking analysis: {num_no_shows} no-show(s) ({no_show_rate:.1%} rate). "
        f"Scheduled rooms={scheduled_rooms}, actual rooms={actual_rooms_used}, "
        f"rooms saved={rooms_saved}."
    )
    return result


# ─── Task 4f: Duration uncertainty ────────────────────────────────────────────

def solve_uncertainty(
    appts_df,
    candidate_rooms,
    duration_buffer_pct,
    keep_no_shows=False,
    solver_msg=False,
    time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
    scenario_name=None,
):
    """
    Task 4f: Inflate appointment durations by duration_buffer_pct and solve.

    Compares inflated result to the uninflated baseline to measure the
    extra room pressure introduced by duration uncertainty.

    extra_rooms_needed = inflated_rooms - base_rooms
    """
    if scenario_name is None:
        pct_label = int(round(duration_buffer_pct * 100))
        scenario_name = f"Uncertainty - {pct_label}% duration buffer"

    appts_df = appts_df.copy()

    # Base solve (no inflation).
    base_result = solve_unrestricted_baseline(
        appts_df=appts_df,
        candidate_rooms=candidate_rooms,
        keep_no_shows=keep_no_shows,
        solver_msg=solver_msg,
        time_limit_seconds=time_limit_seconds,
        scenario_name=f"{scenario_name} [base]",
    )
    base_rooms = base_result["objective_value"]

    # Inflate durations and recompute end times.
    inflated_df = appts_df.copy()
    inflated_df["Appt Duration"] = inflated_df["Appt Duration"] * (1.0 + duration_buffer_pct)
    inflated_df["Appt End"] = (
        inflated_df["Appt Start"] + pd.to_timedelta(inflated_df["Appt Duration"], unit="m")
    )

    inflated_result = solve_unrestricted_baseline(
        appts_df=inflated_df,
        candidate_rooms=candidate_rooms,
        keep_no_shows=keep_no_shows,
        solver_msg=solver_msg,
        time_limit_seconds=time_limit_seconds,
        scenario_name=scenario_name,
    )
    inflated_rooms = inflated_result["objective_value"]
    extra_rooms_needed = (
        (inflated_rooms - base_rooms)
        if inflated_rooms is not None and base_rooms is not None
        else None
    )

    result = inflated_result.copy()
    result["scenario_name"] = scenario_name
    result["policy_name"] = "uncertainty_duration_buffer"
    result["duration_buffer_pct"] = duration_buffer_pct
    result["base_rooms"] = base_rooms
    result["extra_rooms_needed"] = extra_rooms_needed
    result["base_result"] = base_result
    result["notes"] = (
        f"Uncertainty analysis: {int(round(duration_buffer_pct * 100))}% duration buffer. "
        f"Base rooms={base_rooms}, inflated rooms={inflated_rooms}, "
        f"extra rooms needed={extra_rooms_needed}."
    )
    return result


def run_scenario(
    csv_path,
    candidate_rooms,
    keep_no_shows,
    policy,
    room_clusters=None,
    max_rooms_per_provider_day=None,
    blocked_days=None,
    admin_buffer_minutes=None,
    duration_buffer_pct=None,
    no_show_rate=None,
    solver_msg=False,
    time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
    scenario_name=None,
):
    """
    Run one scenario from raw CSV input and return a rich result dictionary.

    Supported policies:
    - "baseline"
    - "policy_a_one_room_per_day"
    - "policy_b_one_cluster_per_day"
    - "k_rooms_per_day"
    - "two_stage"
    - "blocked_days"       (requires blocked_days)
    - "admin_buffer"       (requires admin_buffer_minutes)
    - "overbooking"        (requires no_show_rate)
    - "uncertainty"        (requires duration_buffer_pct)
    """
    # These policies have special loading needs — handle before the default load.
    if policy == "two_stage":
        return solve_two_stage(
            csv_path=csv_path,
            candidate_rooms=candidate_rooms,
            keep_no_shows=keep_no_shows,
            solver_msg=solver_msg,
            time_limit_seconds=time_limit_seconds,
        )

    if policy == "overbooking":
        if no_show_rate is None:
            raise ValueError("no_show_rate must be provided for overbooking.")
        appts_with = load_and_prepare_appointments(csv_path, keep_no_shows=True)
        appts_without = load_and_prepare_appointments(csv_path, keep_no_shows=False)
        return solve_overbooking(
            appts_df_with_noshows=appts_with,
            appts_df_without_noshows=appts_without,
            candidate_rooms=candidate_rooms,
            no_show_rate=no_show_rate,
            solver_msg=solver_msg,
            time_limit_seconds=time_limit_seconds,
            scenario_name=scenario_name or "Overbooking - scheduled vs realized rooms",
        )

    appts_df = load_and_prepare_appointments(csv_path, keep_no_shows=keep_no_shows)

    if policy == "baseline":
        return solve_unrestricted_baseline(
            appts_df=appts_df,
            candidate_rooms=candidate_rooms,
            keep_no_shows=keep_no_shows,
            solver_msg=solver_msg,
            time_limit_seconds=time_limit_seconds,
            scenario_name=scenario_name or "Baseline - Minimize rooms used",
        )

    if policy == "policy_a_one_room_per_day":
        return solve_one_provider_one_room_per_day(
            appts_df=appts_df,
            candidate_rooms=candidate_rooms,
            keep_no_shows=keep_no_shows,
            solver_msg=solver_msg,
            time_limit_seconds=time_limit_seconds,
            scenario_name=scenario_name or "Policy A - One provider, one room/day",
        )

    if policy == "policy_b_one_cluster_per_day":
        if room_clusters is None:
            raise ValueError("room_clusters must be provided for policy_b_one_cluster_per_day.")
        return solve_provider_cluster_per_day(
            appts_df=appts_df,
            candidate_rooms=candidate_rooms,
            room_clusters=room_clusters,
            keep_no_shows=keep_no_shows,
            solver_msg=solver_msg,
            time_limit_seconds=time_limit_seconds,
            scenario_name=scenario_name or "Policy B - Provider assigned to one room cluster/day",
        )

    if policy == "k_rooms_per_day":
        if max_rooms_per_provider_day is None:
            raise ValueError("max_rooms_per_provider_day must be provided for k_rooms_per_day.")
        return solve_provider_room_cap_per_day(
            appts_df=appts_df,
            candidate_rooms=candidate_rooms,
            max_rooms_per_provider_day=max_rooms_per_provider_day,
            keep_no_shows=keep_no_shows,
            solver_msg=solver_msg,
            time_limit_seconds=time_limit_seconds,
            scenario_name=scenario_name
            or f"Policy C - Provider <= {max_rooms_per_provider_day} rooms/day",
        )

    if policy == "blocked_days":
        if blocked_days is None:
            raise ValueError("blocked_days must be provided for blocked_days policy.")
        return solve_blocked_days(
            appts_df=appts_df,
            candidate_rooms=candidate_rooms,
            blocked_days=blocked_days,
            keep_no_shows=keep_no_shows,
            solver_msg=solver_msg,
            time_limit_seconds=time_limit_seconds,
            scenario_name=scenario_name or "Blocked days - Provider availability constraints",
        )

    if policy == "admin_buffer":
        if admin_buffer_minutes is None:
            raise ValueError("admin_buffer_minutes must be provided for admin_buffer policy.")
        return solve_admin_buffer(
            appts_df=appts_df,
            candidate_rooms=candidate_rooms,
            admin_buffer_minutes=admin_buffer_minutes,
            keep_no_shows=keep_no_shows,
            solver_msg=solver_msg,
            time_limit_seconds=time_limit_seconds,
            scenario_name=scenario_name,
        )

    if policy == "uncertainty":
        if duration_buffer_pct is None:
            raise ValueError("duration_buffer_pct must be provided for uncertainty policy.")
        return solve_uncertainty(
            appts_df=appts_df,
            candidate_rooms=candidate_rooms,
            duration_buffer_pct=duration_buffer_pct,
            keep_no_shows=keep_no_shows,
            solver_msg=solver_msg,
            time_limit_seconds=time_limit_seconds,
            scenario_name=scenario_name,
        )

    raise ValueError(f"Unknown policy '{policy}'.")


# ─── Greedy Heuristic: Interval Graph Coloring ────────────────────────────────

def solve_greedy_heuristic(
    appts_df,
    candidate_rooms,
    keep_no_shows=False,
    scenario_name="Greedy Heuristic – Interval Graph Coloring",
):
    """
    Greedy interval-graph coloring heuristic for room assignment.

    Algorithm:
    - Process appointments day by day.
    - Within each day, sort appointments by start time (earliest-start-first).
    - Assign each appointment to the first available room (one whose last
      assigned appointment has already ended by the time the current one starts).
    - If no room is free, open a new room.

    This is a classical O(n log n) greedy algorithm that is known to be optimal
    for pure interval scheduling (no additional constraints). It serves as a
    fast lower-bound benchmark and motivates why MIP is needed when policy
    constraints (clusters, room caps, provider continuity) are layered on top.

    Returns a result dict in the same format as the MIP solvers so it can be
    included directly in comparison tables and charts.
    """
    appts_df = appts_df.copy()
    overlap_pairs = build_overlap_pairs(appts_df)
    same_provider_overlap_pairs = build_same_provider_overlap_pairs(appts_df)

    all_candidate_rooms = list(candidate_rooms)
    assignments = []  # list of {appt_id, Assigned Room}
    rooms_used_global = set()

    for day, day_group in appts_df.groupby("day_str"):
        # Sort by start time; break ties by longest-duration-first (reduces fragmentation).
        day_sorted = day_group.sort_values(
            ["Appt Start", "Appt Duration"], ascending=[True, False]
        )

        # room_end_time[room] = the earliest time the room is free again.
        room_end_time = {}  # room_name -> pd.Timestamp

        for _, row in day_sorted.iterrows():
            appt_start = row["Appt Start"]
            appt_end = row["Appt End"]
            assigned = None

            # Find the first room that is free at appt_start.
            for room in all_candidate_rooms:
                if room not in room_end_time or room_end_time[room] <= appt_start:
                    assigned = room
                    room_end_time[room] = appt_end
                    break

            # If no candidate room is free, we cannot assign (should not happen
            # with 16 rooms and realistic data, but handle gracefully).
            if assigned is None:
                assigned = f"OVERFLOW_{len(room_end_time) + 1}"

            rooms_used_global.add(assigned)
            assignments.append({"appt_id": row["appt_id"], "Assigned Room": assigned})

    assign_df = pd.DataFrame(assignments)
    assignments_df = appts_df.merge(assign_df, on="appt_id", how="left")

    rooms_used = sorted(
        [r for r in rooms_used_global if not r.startswith("OVERFLOW")],
        key=lambda r: int(r.split()[-1]) if r.split()[-1].isdigit() else 999,
    )
    overflow_rooms = [r for r in rooms_used_global if r.startswith("OVERFLOW")]
    rooms_used += overflow_rooms
    n_rooms = len(rooms_used_global)

    result = _base_result_dict(
        scenario_name,
        "greedy_heuristic_interval_coloring",
        keep_no_shows,
        appts_df,
        overlap_pairs,
        same_provider_overlap_pairs,
    )
    result["solver_status"] = "Heuristic"
    result["feasible"] = True
    result["objective_value"] = n_rooms
    result["used_rooms"] = rooms_used
    result["assignments_df"] = assignments_df
    result["provider_day_room_usage_df"] = build_provider_day_room_summary(assignments_df)
    result["notes"] = (
        f"Greedy earliest-start-first interval coloring. "
        f"Rooms used = {n_rooms}. No solver invoked — O(n log n) heuristic. "
        f"Does not enforce policy constraints (clusters, provider caps, etc.)."
    )
    return result


def build_comparison_df(results):
    """Create a compact scenario comparison table."""
    rows = []
    for result in results:
        rows.append(
            {
                "Scenario": result["scenario_name"],
                "Policy": result["policy_name"],
                "No-shows kept?": result["keep_no_shows"],
                "Feasible?": result["feasible"],
                "Rooms used": result["objective_value"],
                "Appointments": result["num_appointments"],
                "Overlap pairs": result["num_overlap_pairs"],
                "Same-provider overlaps": result["num_same_provider_overlap_pairs"],
                # Stage 2 / refinement
                "r_star": result.get("r_star"),
                # Admin buffer
                "Admin buffer (min)": result.get("admin_buffer_minutes"),
                "Near admin window": result.get("num_near_admin_window"),
                # Overbooking
                "No-show rate": result.get("no_show_rate"),
                "No-shows": result.get("num_no_shows"),
                "Scheduled rooms": result.get("scheduled_rooms"),
                "Actual rooms": result.get("actual_rooms_used"),
                "Rooms saved": result.get("rooms_saved"),
                # Uncertainty
                "Duration buffer %": result.get("duration_buffer_pct"),
                "Base rooms": result.get("base_rooms"),
                "Extra rooms needed": result.get("extra_rooms_needed"),
                "Notes": result["notes"],
            }
        )
    return pd.DataFrame(rows)


def run_standard_comparison(
    csv_path,
    candidate_rooms,
    solver_msg=False,
    time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
):
    """
    Recommended starter comparison for validation.

    Scenarios:
    - baseline with no-shows kept
    - baseline with no-shows removed
    - one room/day with no-shows kept
    - one room/day with no-shows removed
    - k=2 with no-shows kept
    - k=2 with no-shows removed
    - k=3 with no-shows kept
    - k=3 with no-shows removed
    """
    scenario_specs = [
        {"scenario_name": "Baseline - Minimize rooms used (keep no-shows)", "keep_no_shows": True, "policy": "baseline"},
        {"scenario_name": "Baseline - Minimize rooms used (remove no-shows)", "keep_no_shows": False, "policy": "baseline"},
        {"scenario_name": "Policy A - One provider, one room/day (keep no-shows)", "keep_no_shows": True, "policy": "policy_a_one_room_per_day"},
        {"scenario_name": "Policy A - One provider, one room/day (remove no-shows)", "keep_no_shows": False, "policy": "policy_a_one_room_per_day"},
        {"scenario_name": "Policy C - k=2 (keep no-shows)", "keep_no_shows": True, "policy": "k_rooms_per_day", "max_rooms_per_provider_day": 2},
        {"scenario_name": "Policy C - k=2 (remove no-shows)", "keep_no_shows": False, "policy": "k_rooms_per_day", "max_rooms_per_provider_day": 2},
        {"scenario_name": "Policy C - k=3 (keep no-shows)", "keep_no_shows": True, "policy": "k_rooms_per_day", "max_rooms_per_provider_day": 3},
        {"scenario_name": "Policy C - k=3 (remove no-shows)", "keep_no_shows": False, "policy": "k_rooms_per_day", "max_rooms_per_provider_day": 3},
    ]

    results = []
    for spec in scenario_specs:
        result = run_scenario(
            csv_path=csv_path,
            candidate_rooms=candidate_rooms,
            keep_no_shows=spec["keep_no_shows"],
            policy=spec["policy"],
            max_rooms_per_provider_day=spec.get("max_rooms_per_provider_day"),
            solver_msg=solver_msg,
            time_limit_seconds=time_limit_seconds,
            scenario_name=spec["scenario_name"],
        )
        results.append(result)

    comparison_df = build_comparison_df(results)
    return results, comparison_df