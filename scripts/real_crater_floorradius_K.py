from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REAL_CSV = ROOT / "C_outputs_PSR_morphometry_002" / "PSR_crater_morphometry_001.csv"
RADIUS = 500.0
CALIBRATION_CSV = f"floor_radius_calibration_R{RADIUS:g}m.csv"
OUTPUT_CSV = f"floor_radius_K_R{RADIUS:g}m_real.csv"
OUTPUT_PNG = f"floor_radius_K_R{RADIUS:g}m_real.png"

def func(x, y0, tau, m):
    return y0 * (np.exp(-x / tau) + m * x)

def sigma_model(y, alpha, c):
    return np.exp(c / 2) * np.maximum(y, 1e-12) ** (alpha / 2)

def roots_for_branch(y_obs, x_grid, curve):
    delta = curve - y_obs
    roots = []

    for i in range(len(x_grid) - 1):
        if delta[i] == 0:
            roots.append(x_grid[i])
        elif delta[i] * delta[i + 1] < 0:
            roots.append(
                x_grid[i]
                - delta[i] * (x_grid[i + 1] - x_grid[i])
                / (delta[i + 1] - delta[i])
            )

    if delta[-1] == 0:
        roots.append(x_grid[-1])

    return np.array(roots)

cal = pd.read_csv(CALIBRATION_CSV).iloc[0]
params = (cal.y0, cal.tau, cal.m)
real = pd.read_csv(REAL_CSV)
y_real = real["floor_radius_over_R"].to_numpy(dtype=float)

y_real_err = (
    real["floor_radius_std"].to_numpy(dtype=float)
    / real["fitted_radius_m"].to_numpy(dtype=float)
)

x_early = np.linspace(0, np.nextafter(float(cal.K_turn), 0.0), 3000)
y_early = func(x_early, *params)
sigma_early = sigma_model(y_early, cal.alpha, cal.c)

results = real.copy()
results["floor_radius_over_R"] = y_real

for label, curve in [
    ("best", y_early),
    ("68_lower_boundary", y_early - sigma_early),
    ("68_upper_boundary", y_early + sigma_early),
    ("95_lower_boundary", y_early - 1.96 * sigma_early),
    ("95_upper_boundary", y_early + 1.96 * sigma_early),
]:
    values = []
    for y in y_real:
        roots = roots_for_branch(y, x_early, curve) if np.isfinite(y) else np.array([])
        values.append(float(roots[0]) if len(roots) == 1 else np.nan)
    results[f"K_over_R2_early_{label}"] = values

for level in ("68", "95"):
    a = results[f"K_over_R2_early_{level}_lower_boundary"].to_numpy()
    b = results[f"K_over_R2_early_{level}_upper_boundary"].to_numpy()
    results[f"K_over_R2_early_{level}_low"] = np.minimum(a, b)
    results[f"K_over_R2_early_{level}_high"] = np.maximum(a, b)

results.to_csv(OUTPUT_CSV, index=False)

plt.figure(figsize=(9, 6))

xx = np.linspace(0, cal.K_max, 1200)
yfit = func(xx, *params)
sigma = sigma_model(yfit, cal.alpha, cal.c)

plt.fill_between(
    xx,
    np.maximum(yfit - 1.96 * sigma, 0),
    yfit + 1.96 * sigma,
    alpha=0.15,
    label="95% interval"
)

plt.fill_between(
    xx,
    np.maximum(yfit - sigma, 0),
    yfit + sigma,
    alpha=0.25,
    label="68% interval"
)

plt.plot(
    xx, yfit,
    color="black",
    linewidth=2,
    label="Calibration fit"
)

plt.errorbar(
    results["K_over_R2_early_best"],
    y_real,
    yerr=y_real_err,
    fmt="o",
    markersize=4,
    capsize=2,
    linestyle="none",
    label="Real craters"
)

plt.axvline(
    cal.K_turn,
    linestyle=":",
    color="gray",
    label="Turning point"
)

plt.xlabel(r"Degradation state $K/R^2$")
plt.ylabel(r"Floor radius $r_{\rm floor}/R_{\rm measured}$")
plt.title("Floor radius vs. degradation state")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()

print("\nReal crater K/R^2:")

for _, row in results.iterrows():
    crater = row.get("PSR_NAME", row.get("CRATER_ID", "Crater"))

    print(
        f"{crater}: "
        f"early={row['K_over_R2_early_best']:.6f}"
    )
print(f"\nSaved results to: {OUTPUT_CSV}")
print(f"Saved plot to: {OUTPUT_PNG}")
print("NaN means no unique crossing before the turning point.")
print("Early-branch solutions are assumed, not independently identified.")