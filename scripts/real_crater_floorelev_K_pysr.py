from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

real_csv = DATA_DIR / "real_crater_morphometry_majorpsr_test.csv"
calibration_csv = DATA_DIR / "floor_calibration_complexity6_mean_004.csv"
output_csv = DATA_DIR / "real_floor_K_majorpsrs_c6_mean_004.csv"
output_png = PLOT_DIR / "real_floor_K_majorpsrs_c6_mean_004.png"

def func(x, A, a):
    return A * np.exp(-a * x)

def sigma_model(y, alpha, c):
    return np.exp(c / 2) * y ** (alpha / 2)

def invert_curve(y, curve, x_grid):
    if not np.isfinite(y) or np.any(np.diff(curve) > 1e-10):
        return np.nan
    if y < curve.min() or y > curve.max():
        return np.nan
    return np.interp(y, curve[::-1], x_grid[::-1])

cal = pd.read_csv(calibration_csv).iloc[0]
A, a = float(cal["A"]), float(cal["a"])
A_err, a_err = float(cal["A_err"]), float(cal["a_err"])
alpha, c = float(cal["alpha"]), float(cal["c"])
x_min, x_max = float(cal["x_min"]), float(cal["x_max"])
if A <= 0 or a <= 0 or not (0 <= x_min < x_max):
    raise ValueError("Calibration parameters or range are invalid.")

x_fit = np.linspace(x_min, x_max, 3000)
y_fit = func(x_fit, A, a)
s = sigma_model(y_fit, alpha, c)
curves = {
    68: (np.maximum(y_fit - s, 0), y_fit + s),
    95: (np.maximum(y_fit - 1.96 * s, 0), y_fit + 1.96 * s),}

real = pd.read_csv(real_csv)
y_real = np.abs(real["floor_elevation_over_R"].to_numpy(dtype=float))
with np.errstate(divide="ignore", invalid="ignore"):
    x_real = -np.log(y_real / A) / a
    y_real_err = real["floor_elevation_std"].to_numpy(dtype=float) / real["fitted_radius_m"].to_numpy(dtype=float)
K_best = np.where(
    np.isfinite(y_real) & (y_real > 0) & np.isfinite(x_real)
    & (x_real >= x_min) & (x_real <= x_max), x_real, np.nan)

results = real.copy()
results["floor_depth_over_R"] = y_real
results["K_over_R2_best"] = K_best
for level, (low, high) in curves.items():
    endpoints = np.array([
        [invert_curve(y, low, x_fit), invert_curve(y, high, x_fit)]
        for y in y_real ])
    both = np.all(np.isfinite(endpoints), axis=1)
    results[f"K_over_R2_{level}_low"] = np.where(both, np.min(endpoints, axis=1), np.nan)
    results[f"K_over_R2_{level}_high"] = np.where(both, np.max(endpoints, axis=1), np.nan)
results["within_calibration_range"] = np.isfinite(K_best)
results.to_csv(output_csv, index=False)

plt.figure(figsize=(8, 6))
plt.fill_between(x_fit, *curves[95], color="lightblue", alpha=0.4, label="1.96σ scatter band")
plt.fill_between(x_fit, *curves[68], color="lightpink", alpha=0.5, label="1σ scatter band")
plt.plot(x_fit, y_fit, "k-", lw=2, label="Complexity-6 fit")
plt.errorbar(K_best, y_real, yerr=y_real_err, fmt="o", color="red", markersize=5,
             capsize=2, label="South Pole crater data")
label = (
    r"$|h_{\mathrm{floor}}|/R=Ae^{-a(K/R^2)}$" "\n"
    rf"$A={A:.6f}\pm{A_err:.6f}$" "\n"
    rf"$a={a:.6f}\pm{a_err:.6f}$")
plt.text(0.98, 0.98, label, transform=plt.gca().transAxes,
         ha="right", va="top", fontsize=11,
         bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
plt.xlabel(r"Degradation state $K/R^2$")
plt.ylabel(r"Floor depth $|h_{\mathrm{floor}}|/R_{\mathrm{meas}}$")
plt.title("Real crater floor-depth degradation")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.close()

for _, row in results.iterrows():
    print(
        f"{row['PSR_NAME']}: {row['K_over_R2_best']:.6f} "
        f"(68% {row['K_over_R2_68_low']:.6f}–{row['K_over_R2_68_high']:.6f}; "
        f"95% {row['K_over_R2_95_low']:.6f}–{row['K_over_R2_95_high']:.6f})"
    )
print(f"Outside calibrated mean curve: {(~results['within_calibration_range']).sum()} of {len(results)}")
print(f"Saved {output_csv} and {output_png}")