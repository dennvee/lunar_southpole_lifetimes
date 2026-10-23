from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PLOT_DIR = ROOT / "plots"

nominal_radius = 500.0

csv_files = [
    "morphometry_vs_K_rim_35m.csv",
    "morphometry_vs_K_rim_40m.csv",
    "morphometry_vs_K_rim_50m.csv",
    "morphometry_vs_K_rim_60m.csv"]

output_png = PLOT_DIR / f"curve_fit_rim_simulations_R{nominal_radius:g}m.png"
variance_png = PLOT_DIR / f"rim_variance_model_R{nominal_radius:g}m.png"
calibration_csv = DATA_DIR / f"rim_calibration_R{nominal_radius:g}m.csv"

def func(x, y0, a, p):
    return y0 * (a / (x + a)) ** p

def sigma_model(yfit, alpha, c):
    return np.exp(c / 2.0) * yfit ** (alpha / 2.0)

def heteroskedastic_fit(x, y, p0):
    popt, pcov = curve_fit(
        func, x, y, p0=p0,
        bounds=([1e-8, 1e-8, 0.0], [np.inf, np.inf, np.inf]),
        maxfev=20000)

    yfit = func(x, *popt)
    residuals = y - yfit

    valid = (
        (yfit > 0)
        & np.isfinite(yfit)
        & np.isfinite(residuals)
        & (residuals != 0))

    alpha, c = np.polyfit(
        np.log(yfit[valid]),
        np.log(residuals[valid] ** 2), 1)

    return popt, pcov, alpha, c

datasets = []

for csv_file in csv_files:
    df = pd.read_csv(DATA_DIR / csv_file)
    x = df["K_over_R2"].to_numpy()
    radius = df["radius"].to_numpy()
    rim_height = df["rim_height"].to_numpy()
    rim_height_std = df["rim_height_std"].to_numpy()
    y = rim_height / radius
    yerr = rim_height_std / radius
    label = csv_file.split("_rim_")[1].split("m")[0] + " m"
    datasets.append((x, y, yerr, label))

x_all = np.concatenate([x for x, _, _, _ in datasets])
y_all = np.concatenate([y for _, y, _, _ in datasets])
y0_guess = np.mean([y[0] for _, y, _, _ in datasets])

popt, pcov, alpha, c = heteroskedastic_fit(
    x_all, y_all, p0=[y0_guess, 0.06, 2.0])

y0, a, p = popt
y0_err, a_err, p_err = np.sqrt(np.diag(pcov))

y_pred_all = func(x_all, y0, a, p)
residuals_y = y_all - y_pred_all
residuals_sq = residuals_y ** 2
variance_y = np.exp(c) * y_pred_all ** alpha

x_fit = np.linspace(x_all.min(), x_all.max(), 1000)
y_fit = func(x_fit, y0, a, p)

pd.DataFrame([{
    "y0": y0,
    "a": a,
    "p": p,
    "y0_err": y0_err,
    "a_err": a_err,
    "p_err": p_err,
    "alpha": alpha,
    "c": c
}]).to_csv(calibration_csv, index=False)

plt.figure(figsize=(8, 6))
colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(datasets)))

for (x, y, yerr, label), color in zip(datasets, colors):
    plt.errorbar(
        x, y, yerr=yerr,
        fmt=".", markersize=4, capsize=2,
        color=color, label=f"{label} simulation")

plt.plot(
    x_fit, y_fit, color="black", linewidth=2,
    label=(
        fr"Fit: "
        fr"$y_0={y0:.4f}\pm{y0_err:.4f}$, "
        fr"$a={a:.4f}\pm{a_err:.4f}$, "
        fr"$p={p:.3f}\pm{p_err:.3f}$"))

plt.xlabel(r"Degradation state $K/R^2$")
plt.ylabel(r"Rim height $h_{\rm rim}/R_{\rm meas}$")
plt.title("Rim height vs. degradation state")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.close()

order = np.argsort(y_pred_all)

plt.figure(figsize=(7, 6))
plt.scatter(
    y_pred_all, residuals_sq,
    s=10, alpha=0.3,
    label=(
        r"$[(h_{\rm rim}/R)_{\rm fit}"
        r"-(h_{\rm rim}/R)_{\rm meas}]^2$"))

plt.plot(
    y_pred_all[order], variance_y[order],
    color="black", linewidth=2,
    label=(
        fr"$\sigma_y^2="
        fr"e^{{{c:.3f}}}"
        fr"\hat{{y}}^{{{alpha:.3f}}}$"))

plt.xlabel(
    r"Fitted rim height $(h_{\rm rim}/R_{\rm meas})_{\rm fit}$")
plt.ylabel(
    r"$[(h_{\rm rim}/R_{\rm meas})_{\rm fit}"
    r"-(h_{\rm rim}/R_{\rm meas})_{\rm meas}]^2$")
plt.title("Heteroskedastic variance in rim height")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(variance_png, dpi=300)
plt.close()

print("\nInitial h_rim/R values:")
for _, y, _, label in datasets:
    print(f"{label}: {y[0]:.6f}")

print("\nHeteroskedastic fit:")
print(f"y0 = {y0:.6f} +/- {y0_err:.6f}")
print(f"a = {a:.6f} +/- {a_err:.6f}")
print(f"p = {p:.6f} +/- {p_err:.6f}")
print(f"alpha = {alpha:.6f}")
print(f"c = {c:.6f}")

print(f"\nSaved calibration plot to: {output_png}")
print(f"Saved variance plot to: {variance_png}")
print(f"Saved calibration parameters to: {calibration_csv}")