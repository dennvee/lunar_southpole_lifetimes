from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
input_csv = ROOT / "data" / "rim_lifetimes_majorpsrs_complexity5_004.csv"
output_png = ROOT / "plots" / "rim_lifetimes_majorpsrs_complexity5_004.png"
output_png.parent.mkdir(parents=True, exist_ok=True)

real = pd.read_csv(input_csv)
real = real[np.isfinite(real["t_best_Myr"]) & (real["t_best_Myr"] >= 0)].copy()
real = real.sort_values("t_best_Myr").reset_index(drop=True)
if real.empty:
    raise ValueError("No finite best-fit lifetimes to plot.")

names = real["PSR_NAME"].astype(str).to_numpy() if "PSR_NAME" in real else real.index.astype(str).to_numpy()
ages = real["t_best_Myr"].to_numpy(dtype=float) / 1000
y = np.arange(len(real))
extrap = real["kappa_extrapolated"].fillna(False).astype(bool).to_numpy() if "kappa_extrapolated" in real else np.zeros(len(real), dtype=bool)

fig, ax = plt.subplots(figsize=(10, max(6, 0.36 * len(real) + 1.8)))
for prefix, color, width, z in [("95", "#f8bbd0", 2.0, 1), ("68", "#ec407a", 4.0, 2)]:
    low = real[f"t_{prefix}_low_Myr"].to_numpy(dtype=float) / 1000
    high = real[f"t_{prefix}_high_Myr"].to_numpy(dtype=float) / 1000
    valid = np.isfinite(low) & np.isfinite(high) & (low <= ages) & (ages <= high)
    ax.hlines(y[valid], low[valid], high[valid], color=color, linewidth=width, zorder=z)

ax.scatter(ages[~extrap], y[~extrap], s=36, color="#880e4f", zorder=3)
ax.scatter(ages[extrap], y[extrap], s=45, facecolors="white", edgecolors="#880e4f", linewidths=1.5, zorder=4)
ax.set_yticks(y, names)
ax.set_xlabel("Inferred crater lifetime (Gyr)")
ax.set_ylabel("PSR crater")
ax.set_title("South polar crater lifetimes — complexity-5 rim model")
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)
ax.set_xlim(left=0)
ax.legend(handles=[
    Line2D([0], [0], color="#f8bbd0", lw=2, label="95% model-scatter interval"),
    Line2D([0], [0], color="#ec407a", lw=4, label="68% model-scatter interval"),
    Line2D([0], [0], marker="o", linestyle="none", color="#880e4f", label="Best-fit lifetime"),
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#880e4f", label="Extrapolated Trask κ"),
], fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(output_png, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Plotted {len(real)} craters; {extrap.sum()} require extrapolated kappa.")
print(f"Saved {output_png}")
