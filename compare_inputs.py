"""
compare_inputs.py
-----------------
Compares the appointment DataFrames produced by the two loading pipelines:

  1. clinic_room_assignment.load_and_prepare_appointments()   (legacy)
  2. clinic_scheduler.data_loader.load_appointments()         (column-gen)

Prints row counts, a drop-reason breakdown for each pipeline, and any
patient/appointment keys present in one pipeline but not the other.

No solvers are invoked.
"""

import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
CSV_PATH = "/Users/eileenerkan/Desktop/435_Project/AppointmentDataWeek1.csv"

# ── Imports ───────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from clinic_room_assignment import load_and_prepare_appointments
from clinic_scheduler.data_loader import load_appointments


def drop_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Walk through each drop step on the raw CSV and return a summary DataFrame
    showing how many rows each filter removes.
    """
    raw = raw.copy()
    raw.columns = [c.strip() for c in raw.columns]
    for col in raw.columns:
        if raw[col].dtype == "object":
            raw[col] = raw[col].astype(str).str.strip()

    steps = []
    total = len(raw)
    steps.append(("Raw rows loaded", total, 0))

    after = raw[raw["Cancelled Appts"] != "Y"] if "Cancelled Appts" in raw.columns else raw
    steps.append(("Drop Cancelled=Y", len(after), total - len(after)))
    raw = after

    after = raw[raw["Deleted Appts"] != "Y"] if "Deleted Appts" in raw.columns else raw
    steps.append(("Drop Deleted=Y", len(after), total - len(after)))
    total = len(raw)
    raw = after

    after = raw[raw["No Show Appts"] != "Y"] if "No Show Appts" in raw.columns else raw
    steps.append(("Drop No-shows (keep_no_shows=False)", len(after), total - len(after)))
    total = len(raw)
    raw = after

    if "Appt Type" in raw.columns:
        after = raw[raw["Appt Type"].str.upper() != "ADMIN TIME"]
    else:
        after = raw
    steps.append(("Drop ADMIN TIME", len(after), total - len(after)))
    total = len(raw)
    raw = after

    raw["Appt Date"] = pd.to_datetime(raw["Appt Date"], format="%m-%d-%Y", errors="coerce")
    after = raw[raw["Appt Date"] != pd.Timestamp("2025-11-11")]
    steps.append(("Drop 2025-11-11 (Tue, clinic closed)", len(after), total - len(after)))

    return pd.DataFrame(steps, columns=["Step", "Rows remaining", "Rows dropped"])


def make_key(df: pd.DataFrame, date_col: str, provider_col: str,
             patient_col: str, start_col: str, duration_col: str) -> pd.Series:
    """Build a composite key: date|provider|patient|start|duration."""
    return (
        df[date_col].astype(str).str[:10]
        + "|" + df[provider_col].astype(str).str.strip()
        + "|" + df[patient_col].astype(str).str.strip()
        + "|" + df[start_col].astype(str)
        + "|" + df[duration_col].astype(str)
    )


# ── Load raw CSV once for summary ─────────────────────────────────────────────
raw_df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("DROP-REASON SUMMARY  (shared steps, both pipelines apply the same rules)")
print("=" * 70)
summary = drop_summary(raw_df)
print(summary.to_string(index=False))

# ── Load via each pipeline ────────────────────────────────────────────────────
legacy_df = load_and_prepare_appointments(CSV_PATH, keep_no_shows=False)
colgen_df = load_appointments(CSV_PATH)

# clinic_scheduler additionally excludes no-shows at the *solver* level via
# filter_appointments_for_policy, but load_appointments() keeps them flagged.
# For a fair apples-to-apples count, exclude no-shows from colgen too.
colgen_no_ns = colgen_df[~colgen_df["no_show"]].copy()

print()
print("=" * 70)
print("ROW COUNTS")
print("=" * 70)
print(f"  clinic_room_assignment  (legacy, no-shows excluded) : {len(legacy_df):>4} rows")
print(f"  clinic_scheduler        (colgen, no-shows included) : {len(colgen_df):>4} rows")
print(f"  clinic_scheduler        (colgen, no-shows excluded) : {len(colgen_no_ns):>4} rows")

# ── Build composite keys for comparison ──────────────────────────────────────
# legacy uses: Appt Date, Primary Provider, Patient Id, start (from Appt Start), Appt Duration
legacy_df["_key"] = (
    legacy_df["Appt Date"].dt.strftime("%Y-%m-%d")
    + "|" + legacy_df["Primary Provider"].astype(str).str.strip()
    + "|" + legacy_df["Patient Id"].astype(str).str.strip()
    + "|" + legacy_df["Appt Start"].dt.strftime("%H:%M")
    + "|" + legacy_df["Appt Duration"].astype(int).astype(str)
)

colgen_no_ns["_key"] = (
    colgen_no_ns["date"].dt.strftime("%Y-%m-%d")
    + "|" + colgen_no_ns["provider"].astype(str).str.strip()
    + "|" + colgen_no_ns["patient_id"].astype(str).str.strip()
    + "|" + colgen_no_ns["start_min"].apply(lambda m: f"{int(m)//60:02d}:{int(m)%60:02d}")
    + "|" + colgen_no_ns["duration"].astype(int).astype(str)
)

legacy_keys = set(legacy_df["_key"])
colgen_keys = set(colgen_no_ns["_key"])

only_legacy = sorted(legacy_keys - colgen_keys)
only_colgen = sorted(colgen_keys - legacy_keys)

print()
print("=" * 70)
print("KEY DIFFERENCES  (date|provider|patient|start|duration)")
print("=" * 70)

if not only_legacy and not only_colgen:
    print("  Pipelines produce identical appointment sets after no-show exclusion.")
else:
    if only_legacy:
        print(f"\n  In legacy only ({len(only_legacy)} rows):")
        for k in only_legacy:
            print(f"    {k}")
    if only_colgen:
        print(f"\n  In colgen only ({len(only_colgen)} rows):")
        for k in only_colgen:
            print(f"    {k}")

print()
print("=" * 70)
print("PIPELINE DIFFERENCES SUMMARY")
print("=" * 70)
rows = [
    ("Drop Cancelled=Y",                    "Yes", "Yes"),
    ("Drop Deleted=Y",                      "Yes", "Yes"),
    ("Drop No-shows (default)",             "Yes (keep_no_shows=False)", "Flagged, not dropped at load time"),
    ("Drop ADMIN TIME",                     "Yes (after fix)", "Yes"),
    ("Exclude 2025-11-11 (Tue, closed)",    "Yes (after fix)", "Yes (week==1 filter in load_week_inputs)"),
    ("Drop duration <= 0",                  "No (dropna only)", "Yes (explicit duration > 0 check)"),
]
header = f"  {'Rule':<45} {'legacy':^30} {'colgen':^40}"
print(header)
print("  " + "-" * (len(header) - 2))
for rule, leg, cg in rows:
    print(f"  {rule:<45} {leg:<30} {cg}")

print()
