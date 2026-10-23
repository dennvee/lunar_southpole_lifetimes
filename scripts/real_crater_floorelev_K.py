from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REAL_CSV = ROOT / 'data' / 'PSR_crater_morphometry_001.csv'
nominal_radius = 500.0
calibration_csv = ROOT /'data'/f'floor_calibration_R{nominal_radius:g}m.csv'
output_csv = ROOT/'data'/ f'floor_K_heteroskedastic_R{nominal_radius:g}m_real_psrs.csv'
output_png = ROOT/'plots'/f'floor_fit_R{nominal_radius:g}m_real_psrs.png'

def func(x, y0, a, p):
    return y0 * (a / (x + a)) ** p

def sigma_model(y, alpha, c):
    return np.exp(c / 2) * y ** (alpha / 2)

cal = pd.read_csv(calibration_csv).iloc[0]
y0, a, p, alpha, c = [cal[k] for k in ('y0', 'a', 'p', 'alpha', 'c')]
x_fit = np.linspace(0, cal['x_max'], 3000)
y_fit = func(x_fit, y0, a, p)
s = sigma_model(y_fit, alpha, c)
bands = [np.maximum(y_fit - z*s, 0) for z in (1, 1.96)] + [y_fit + z*s for z in (1, 1.96)]

def invert_curve(y, curve):
    if np.any(np.diff(curve) > 1e-10):
        return np.nan
    if y < curve.min() or y > curve.max():
        return np.nan
    return np.interp(y, curve[::-1], x_fit[::-1])

real = pd.read_csv(REAL_CSV)
y_real = np.abs(real['floor_elevation_over_R'].to_numpy(dtype=float))
K_best = np.full(len(real), np.nan)
valid = (y_real > 0) & (y_real <= y0)
K_best[valid] = a * ((y0 / y_real[valid]) ** (1/p) - 1)
results = real.copy()
results['floor_depth_over_R'] = y_real
results['K_over_R2_best'] = K_best
for z, low, high in ((68, bands[0], bands[2]), (95, bands[1], bands[3])):
    endpoints = np.array([[invert_curve(y, low), invert_curve(y, high)] for y in y_real])
    results[f'K_over_R2_{z}_low'] = np.where(np.all(np.isfinite(endpoints), axis=1), np.min(endpoints, axis=1), np.nan)
    results[f'K_over_R2_{z}_high'] = np.where(np.all(np.isfinite(endpoints), axis=1), np.max(endpoints, axis=1), np.nan)
results.to_csv(output_csv, index=False)

plt.figure(figsize=(8, 6))

plt.fill_between(
    x_fit, bands[1], bands[3],
    color="lightblue",
    alpha=0.4,
    label=r"$1.96\sigma$ interval"
)

plt.fill_between(
    x_fit, bands[0], bands[2],
    color="lightpink",
    alpha=0.5,
    label=r"$1\sigma$ interval"
)

plt.plot(
    x_fit, y_fit,
    color="black",
    linewidth=2,
    label="Simulation calibration"
)

equation = r"$\frac{|h_{floor}|}{R}=y_0\left(\frac{a}{K/R^2+a}\right)^p$"

plt.gca().text(
    0.98, 0.98, equation,
    transform=plt.gca().transAxes,
    fontsize=12,
    ha="right",
    va="top",
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray")
)

plt.scatter(
    K_best, y_real,
    color="red",
    s=20,
    label="Real PSR craters"
)

plt.xlabel(r"Degradation state $K/R^2$")
plt.ylabel(r"Floor depth $|h_{floor}|/R_{meas}$")
plt.title("Floor depth vs. degradation state")

plt.legend(
    fontsize=9,
    loc="center",
    bbox_to_anchor=(0.57, 0.78),
    framealpha=0.9
)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.close()
for _, row in results.iterrows():
    print(f"{row['PSR_NAME']}: {row['K_over_R2_best']:.6f} "
          f"(68% {row['K_over_R2_68_low']:.6f}-{row['K_over_R2_68_high']:.6f}; "
          f"95% {row['K_over_R2_95_low']:.6f}-{row['K_over_R2_95_high']:.6f})")
print(f'Saved {output_csv} and {output_png}')