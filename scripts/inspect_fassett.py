
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "catalogues" / "CraterHostsforPSRuniques_v2_reprojected.csv"
OUTPUT = ROOT / "plots" / "fassett_diameter_distribution.png"

df = pd.read_csv(CSV)
diameters = df["diam_m"].dropna().to_numpy()
diameters = diameters[diameters > 0]

bins = np.logspace(np.log10(diameters.min()), np.log10(diameters.max()), 25)
counts, edges = np.histogram(diameters, bins=bins)

fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.RdPu(np.linspace(0.3, 0.85, len(counts)))

ax.bar(
    edges[:-1],
    counts,
    width=np.diff(edges),
    align="edge",
    color=colors,
    edgecolor="white",
    linewidth=0.7
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Crater diameter (m)")
ax.set_ylabel("Number of craters")
ax.set_title("Fassett PSR Crater Diameter Distribution")
ax.grid(alpha=0.2, which="both")
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=300)
plt.close()

print(f"Total craters: {len(diameters)}")
print(f"Diameter range: {diameters.min():.1f}–{diameters.max():.1f} m")

print("\nDiameter bins:")
for left, right, count in zip(edges[:-1], edges[1:], counts):
    print(f"{left:9.1f}–{right:9.1f} m: {count:6d}")

print(f"\nSaved: {OUTPUT}")
