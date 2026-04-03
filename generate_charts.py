"""
generate_charts.py
Generate three Week 1 policy comparison charts using actual KPI values
from the schedule CSVs and policy comparison table.

Charts saved to results/week1/:
  1. policy_coverage_travel.png   — grouped bar: coverage % + travel distance
  2. policy_tradeoff_scatter.png  — scatter: travel vs coverage, colored by conflicts
  3. conflict_comparison.png      — horizontal bars: room conflicts per policy
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path("results/week1")
OUT.mkdir(parents=True, exist_ok=True)

# ── KPI data from schedule CSVs (coverage/travel) and user-provided conflicts ─
# coverage_rate: scheduled real appointments / 427 total input appointments
# travel: computed via dist_matrix from schedule CSVs
# conflicts: from user-authoritative values (Chart 3 description)

DATA = {
    # name                  coverage%  travel(m)  conflicts
    "Historical":          (96.3,      0.0,        159),
    "CG Optimal":          (96.3,      33.2,         0),
    "Policy A":            (96.0,       0.0,       120),
    "Policy B (2m)":       (None,      None,          4),   # not yet run
    "Policy B (3m)":       (96.3,      32.0,         0),
    "Policy C":            (71.2,       8.1,          0),
    "Policy D":            (96.3,      33.2,          0),
    "Policy E":            (91.8,      56.4,          0),
    "Policy F":            (82.4,     138.8,          0),
}

CHART1_ORDER = [
    "Historical", "CG Optimal", "Policy A",
    "Policy B (2m)", "Policy B (3m)",
    "Policy C", "Policy D", "Policy E", "Policy F",
]

CHART3_ORDER = [
    "Historical", "Policy A", "Policy B (2m)",
    "CG Optimal", "Policy B (3m)", "Policy C",
    "Policy D", "Policy E", "Policy F",
]

TEAL   = "#2CA6A4"
ORANGE = "#E8821A"
GREEN  = "#2CA44E"
RED    = "#E03C3C"


# ─── Chart 1: Grouped bar — Coverage % + Travel Distance ─────────────────────

fig, ax1 = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

x     = np.arange(len(CHART1_ORDER))
width = 0.35

cov_vals    = [DATA[p][0] if DATA[p][0] is not None else 0 for p in CHART1_ORDER]
travel_vals = [DATA[p][1] if DATA[p][1] is not None else 0 for p in CHART1_ORDER]
cov_missing    = [DATA[p][0] is None for p in CHART1_ORDER]
travel_missing = [DATA[p][1] is None for p in CHART1_ORDER]

bars1 = ax1.bar(x - width / 2, cov_vals, width, color=TEAL,   zorder=3)
ax1.set_ylabel("Coverage Rate (%)", fontsize=12, color=TEAL)
ax1.tick_params(axis="y", labelcolor=TEAL, labelsize=11)
ax1.set_ylim(0, 140)
ax1.set_xticks(x)
ax1.set_xticklabels(CHART1_ORDER, rotation=20, ha="right", fontsize=11)
ax1.set_xlabel("Policy", fontsize=12)
ax1.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax1.set_axisbelow(True)

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width / 2, travel_vals, width, color=ORANGE, zorder=3)
ax2.set_ylabel("Total Travel Distance (m)", fontsize=12, color=ORANGE)
ax2.tick_params(axis="y", labelcolor=ORANGE, labelsize=11)
ax2.set_ylim(0, max(travel_vals) * 1.5 if max(travel_vals) > 0 else 10)

for bar, val, missing in zip(bars1, cov_vals, cov_missing):
    label = "N/A" if missing else f"{val:.1f}%"
    color = "grey" if missing else TEAL
    y = 3 if missing else bar.get_height() + 1.5
    ax1.text(bar.get_x() + bar.get_width() / 2, y, label,
             ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

for bar, val, missing in zip(bars2, travel_vals, travel_missing):
    label = "N/A" if missing else f"{val:.1f}"
    color = "grey" if missing else ORANGE
    y = 1 if missing else bar.get_height() + 0.5
    ax2.text(bar.get_x() + bar.get_width() / 2, y, label,
             ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

p1 = mpatches.Patch(color=TEAL,   label="Coverage (%)")
p2 = mpatches.Patch(color=ORANGE, label="Travel Distance (m)")
ax1.legend(handles=[p1, p2], fontsize=11, loc="upper right")
ax1.set_title("Week 1: Coverage vs Travel Distance by Policy",
              fontsize=14, fontweight="bold", pad=12)

fig.tight_layout()
fig.savefig(OUT / "policy_coverage_travel.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved policy_coverage_travel.png")


# ─── Chart 2: Scatter — Travel vs Coverage, colored by conflicts ──────────────

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Label nudge offsets to avoid overlap: (dx_pts, dy_pts)
OFFSETS = {
    "Historical":    ( 6,   1),
    "CG Optimal":    ( 6,   1),
    "Policy A":      ( 6,  -3),
    "Policy B (3m)": ( 6,  -3),
    "Policy C":      ( 6,   1),
    "Policy D":      ( 6,  -3),
    "Policy E":      ( 6,   1),
    "Policy F":      ( 6,   1),
}

for name in CHART1_ORDER:
    cov, travel, conflicts = DATA[name]
    if cov is None or travel is None:
        continue
    color = GREEN if conflicts == 0 else RED
    ax.scatter(travel, cov, color=color, s=130, zorder=4,
               edgecolors="white", linewidths=0.8)
    dx, dy = OFFSETS.get(name, (6, 1))
    ax.annotate(name, (travel, cov), xytext=(dx, dy),
                textcoords="offset points", fontsize=10, ha="left")

ax.set_xlabel("Total Travel Distance (m)", fontsize=12)
ax.set_ylabel("Coverage Rate (%)", fontsize=12)
ax.set_title("Coverage vs Travel Distance Tradeoff", fontsize=14, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.tick_params(labelsize=11)

p_green = mpatches.Patch(color=GREEN, label="0 conflicts")
p_red   = mpatches.Patch(color=RED,   label=">0 conflicts")
ax.legend(handles=[p_green, p_red], fontsize=11)

fig.tight_layout()
fig.savefig(OUT / "policy_tradeoff_scatter.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved policy_tradeoff_scatter.png")


# ─── Chart 3: Horizontal bar — Room Conflicts ─────────────────────────────────

conflict_vals = [DATA[p][2] for p in CHART3_ORDER]
bar_colors    = [RED if v > 0 else GREEN for v in conflict_vals]
max_val       = max(conflict_vals)

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars = ax.barh(CHART3_ORDER, conflict_vals, color=bar_colors,
               edgecolor="white", height=0.6, zorder=3)
ax.set_xlabel("Room Conflicts", fontsize=12)
ax.set_title("Room Conflicts by Policy — Week 1", fontsize=14, fontweight="bold")
ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.invert_yaxis()
ax.tick_params(labelsize=11)

for bar, val in zip(bars, conflict_vals):
    color = RED if val > 0 else GREEN
    ax.text(bar.get_width() + max_val * 0.012,
            bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=11, fontweight="bold", color=color)

p_red   = mpatches.Patch(color=RED,   label="Conflicts > 0")
p_green = mpatches.Patch(color=GREEN, label="No conflicts")
ax.legend(handles=[p_red, p_green], fontsize=11, loc="lower right")

fig.tight_layout()
fig.savefig(OUT / "conflict_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved conflict_comparison.png")

print("\nDone. Data used:")
print(f"  {'Policy':<18} {'Coverage':>10} {'Travel':>10} {'Conflicts':>10}")
for name in CHART1_ORDER:
    cov, travel, conflicts = DATA[name]
    print(f"  {name:<18} {str(cov)+'%':>10} {str(travel)+'m':>10} {conflicts:>10}")
