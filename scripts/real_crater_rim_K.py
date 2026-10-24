from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REAL_CSV = ROOT / "data" / "real_crater_morphometry_majorpsrs.csv"
# REAL_CSV = ROOT / "plaw_data" / "plaw_morphometry_for_curvefit.csv"

nominal_radius = 500.0

calibration_csv = ROOT /"data" /f"rim_calibration.csv"
output_png = ROOT /"plots"/ f"real_rim_K_majorpsrs.png"
output_csv = ROOT /"data" /f"real_rim_K_majorpsrs.csv"

def func(x, y0, a, p):
    return y0 * (a / (x + a)) ** p

def invert_func(y, y0, a, p):
    return np.maximum(
        a * ((y0 / y) ** (1.0 / p) - 1.0), 0.0)

def sigma_model(yfit, alpha, c):
    return np.exp(c / 2.0) * yfit ** (alpha / 2.0)

def invert_curve(y_value, curve, x_fit):
    if y_value < curve.min() or y_value > curve.max():
        return np.nan
    return np.interp(y_value, curve[::-1], x_fit[::-1])

calibration = pd.read_csv(calibration_csv).iloc[0]

y0 = calibration["y0"]
a = calibration["a"]
p = calibration["p"]
y0_err = calibration["y0_err"]
a_err = calibration["a_err"]
p_err = calibration["p_err"]
alpha = calibration["alpha"]
c = calibration["c"]

x_fit = np.linspace(0.0, 0.20, 1000)
y_fit = func(x_fit, y0, a, p)
sigma_fit = sigma_model(y_fit, alpha, c)

y68_low = np.maximum(y_fit - sigma_fit, 0.0)
y68_high = y_fit + sigma_fit
y95_low = np.maximum(y_fit - 1.96 * sigma_fit, 0.0)
y95_high = y_fit + 1.96 * sigma_fit

real = pd.read_csv(REAL_CSV)

y_real = real["rim_height_over_R"].to_numpy()
y_real_err = (
    real["rim_height_std"].to_numpy()
    / real["fitted_radius_m"].to_numpy())

K_best = invert_func(y_real, y0, a, p)

K68_a = np.array([
    invert_curve(y, y68_low, x_fit) for y in y_real])
K68_b = np.array([
    invert_curve(y, y68_high, x_fit) for y in y_real])

K95_a = np.array([
    invert_curve(y, y95_low, x_fit) for y in y_real])
K95_b = np.array([
    invert_curve(y, y95_high, x_fit) for y in y_real])

K68_low = np.minimum(K68_a, K68_b)
K68_high = np.maximum(K68_a, K68_b)
K95_low = np.minimum(K95_a, K95_b)
K95_high = np.maximum(K95_a, K95_b)

results = real.copy()
results["K_over_R2_best"] = K_best
results["K_over_R2_68_low"] = K68_low
results["K_over_R2_68_high"] = K68_high
results["K_over_R2_95_low"] = K95_low
results["K_over_R2_95_high"] = K95_high
results.to_csv(output_csv, index=False)

plt.figure(figsize=(8, 6))

plt.fill_between(
    x_fit, y95_low, y95_high,
    color="lightblue",
    alpha=0.4,
    label=r"$1.96\sigma$")

plt.fill_between(
    x_fit, y68_low, y68_high,
    color="lightpink",
    alpha=0.5,
    label=r"$1\sigma$")

plt.plot(
    x_fit, y_fit,
    color="black",
    linewidth=2,
    label=(
        fr"Fit: "
        fr"$y_0={y0:.4f}\pm{y0_err:.4f}$, "
        fr"$a={a:.4f}\pm{a_err:.4f}$, "
        fr"$p={p:.3f}\pm{p_err:.3f}$"))

equation = r"$\frac{f_r}{R}=y_0\left(\frac{a}{K/R^2+a}\right)^p$"

plt.gca().text(
    0.98, 0.98, equation,
    transform=plt.gca().transAxes,
    fontsize=12,
    ha="right", va="top",
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

plt.plot(
    x_fit, y_fit,
    color="black",
    linewidth=2,
    label=(
        fr"Fit: "
        fr"$y_0={y0:.4f}\pm{y0_err:.4f}$, "
        fr"$a={a:.4f}\pm{a_err:.4f}$, "
        fr"$p={p:.3f}\pm{p_err:.3f}$"))

equation = r"$\frac{f_r}{R}=y_0\left(\frac{a}{K/R^2+a}\right)^p$"

plt.gca().text(
    0.98, 0.98, equation,
    transform=plt.gca().transAxes,
    fontsize=12,
    ha="right", va="top",
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

plt.errorbar(
    K_best, y_real, yerr=y_real_err,
    fmt="o", color="red", markersize=5,
    capsize=2, label="South Pole crater data")

plt.xlabel(r"Degradation state $K/R^2$")
plt.ylabel(r"Rim height $h_{\rm rim}/R_{\rm meas}$")
plt.title("Rim height vs. degradation state")
plt.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.98))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.close()

print("\nReal crater K/R^2:")

for _, row in results.iterrows():
    print(
        f"{row['PSR_NAME']}: "
        f"{row['K_over_R2_best']:.6f} "
        f"(68% "
        f"{row['K_over_R2_68_low']:.6f}-"
        f"{row['K_over_R2_68_high']:.6f}; "
        f"95% "
        f"{row['K_over_R2_95_low']:.6f}-"
        f"{row['K_over_R2_95_high']:.6f})")

print(f"\nSaved plot to: {output_png}")
print(f"Saved results to: {output_csv}")