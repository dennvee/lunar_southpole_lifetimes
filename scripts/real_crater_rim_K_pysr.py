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
calibration_csv = DATA_DIR / "rim_calibration_complexity5_test.csv"
output_png = PLOT_DIR / "real_rim_K_majorpsrs_complexity5_test.png"
output_csv = DATA_DIR / "real_rim_K_majorpsrs_complexity5_test.csv"

def func(x, a, b):
    return a / (x + b)

def sigma_model(y, alpha, c):
    return np.exp(c / 2) * y ** (alpha / 2)

def invert_func(y, a, b, x_max):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = a / y - b
    return np.where(np.isfinite(y) & (y > 0) & (x >= 0) & (x <= x_max), x, np.nan)

def invert_curve(y, curve, x_grid):
    if not np.isfinite(y) or y < curve.min() or y > curve.max():
        return np.nan
    return np.interp(y, curve[::-1], x_grid[::-1])

calibration = pd.read_csv(calibration_csv).iloc[0]
a, b = float(calibration["a"]), float(calibration["b"])
a_err, b_err = float(calibration["a_err"]), float(calibration["b_err"])
alpha, c = float(calibration["alpha"]), float(calibration["c"])
x_max = float(calibration["x_max"])

x_fit = np.linspace(0, x_max, 2000)
y_fit = func(x_fit, a, b)
sigma_fit = sigma_model(y_fit, alpha, c)

y68_low = np.maximum(y_fit - sigma_fit, 0)
y68_high = y_fit + sigma_fit
y95_low = np.maximum(y_fit - 1.96 * sigma_fit, 0)
y95_high = y_fit + 1.96 * sigma_fit

real = pd.read_csv(real_csv)
y_real = real["rim_height_over_R"].to_numpy(dtype=float)
radius = real["fitted_radius_m"].to_numpy(dtype=float)
with np.errstate(divide="ignore", invalid="ignore"):
    y_real_err = real["rim_height_std"].to_numpy(dtype=float) / radius

K_best = invert_func(y_real, a, b, x_max)
K68_a = np.array([invert_curve(y, y68_low, x_fit) for y in y_real])
K68_b = np.array([invert_curve(y, y68_high, x_fit) for y in y_real])
K95_a = np.array([invert_curve(y, y95_low, x_fit) for y in y_real])
K95_b = np.array([invert_curve(y, y95_high, x_fit) for y in y_real])

results = real.copy()
results["K_over_R2_best"] = K_best
results["K_over_R2_68_low"] = np.minimum(K68_a, K68_b)
results["K_over_R2_68_high"] = np.maximum(K68_a, K68_b)
results["K_over_R2_95_low"] = np.minimum(K95_a, K95_b)
results["K_over_R2_95_high"] = np.maximum(K95_a, K95_b)
results["within_calibration_range"] = np.isfinite(K_best)
results.to_csv(output_csv, index=False)

plt.figure(figsize=(8, 6))
plt.fill_between(x_fit, y95_low, y95_high, color="lightblue", alpha=0.4, label=r"$1.96\sigma$ scatter band")
plt.fill_between(x_fit, y68_low, y68_high, color="lightpink", alpha=0.5, label=r"$1\sigma$ scatter band")
plt.plot(x_fit, y_fit, color="black", linewidth=2, label="Complexity-5 fit")
plt.errorbar(K_best, y_real, yerr=y_real_err, fmt="o", color="red", markersize=5,
             capsize=2, label="South Pole crater data")
equation = (
    r"$\frac{h_{\mathrm{rim}}}{R_{\mathrm{meas}}}=\frac{a}{K/R^2+b}$" "\n"
    rf"$a={a:.6f}\pm{a_err:.6f}$" "\n"
    rf"$b={b:.6f}\pm{b_err:.6f}$"
)
plt.gca().text(0.98, 0.98, equation, transform=plt.gca().transAxes,
               ha="right", va="top", fontsize=11,
               bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
plt.xlabel(r"Degradation state $K/R^2$")
plt.ylabel(r"Rim height $h_{\mathrm{rim}}/R_{\mathrm{meas}}$")
plt.title("Real crater rim-height degradation — complexity 5")
plt.legend(fontsize=8, loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.close()

print("\nReal crater K/R^2:")
for _, row in results.iterrows():
    print(
        f"{row['PSR_NAME']}: {row['K_over_R2_best']:.6f} "
        f"(68% {row['K_over_R2_68_low']:.6f}–{row['K_over_R2_68_high']:.6f}; "
        f"95% {row['K_over_R2_95_low']:.6f}–{row['K_over_R2_95_high']:.6f})"
    )

n_out = (~results["within_calibration_range"]).sum()
print(f"\nCrater measurements outside the calibrated mean curve: {n_out} of {len(results)}")
print(f"Saved plot to: {output_png}")
print(f"Saved results to: {output_csv}")
