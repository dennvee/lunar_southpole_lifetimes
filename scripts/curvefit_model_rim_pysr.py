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
PLOT_DIR.mkdir(exist_ok=True)

heights = [35, 40, 50, 60]
calibration_csv = DATA_DIR / "rim_calibration_complexity5_003.csv"
output_png = PLOT_DIR / "curve_fit_rim_complexity5_003.png"
variance_png = PLOT_DIR / "rim_variance_complexity5_003.png"


def func(x, a, b):
    return a / (x + b)

def sigma_model(y, alpha, c):
    return np.exp(c / 2) * y ** (alpha / 2)

datasets = []
for h in heights:
    df = pd.read_csv(DATA_DIR / f"morphometry_vs_K_rim_{h}m_renormalized_003.csv")
    x = df["K_over_R2"].to_numpy(dtype=float)
    radius = df["radius"].to_numpy(dtype=float)
    y = df["rim_height"].to_numpy(dtype=float) / radius
    yerr = df["rim_height_std"].to_numpy(dtype=float) / radius
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (radius > 0) & (x >= 0) & (y > 0)
    datasets.append((x[valid], y[valid], yerr[valid], f"{h} m"))

x_all = np.concatenate([d[0] for d in datasets])
y_all = np.concatenate([d[1] for d in datasets])
popt, pcov = curve_fit(
    func, x_all, y_all,
    p0=[0.0028671278, 0.031987514],
    bounds=([1e-12, 1e-12], [np.inf, np.inf]),
    maxfev=30000,
)
a, b = popt
a_err, b_err = np.sqrt(np.diag(pcov))

yhat = func(x_all, *popt)
residuals = y_all - yhat
valid = (yhat > 0) & np.isfinite(residuals) & (np.abs(residuals) > 1e-12)
alpha, c = np.polyfit(np.log(yhat[valid]), np.log(residuals[valid] ** 2), 1)

pd.DataFrame([{
    "a": a, "b": b, "a_err": a_err, "b_err": b_err,
    "alpha": alpha, "c": c, "x_max": x_all.max(),
}]).to_csv(calibration_csv, index=False)

x_fit = np.linspace(0, x_all.max(), 1000)
y_fit = func(x_fit, *popt)
s = sigma_model(y_fit, alpha, c)

plt.figure(figsize=(8, 6))
colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(datasets)))
for (x, y, yerr, label), color in zip(datasets, colors):
    plt.errorbar(x, y, yerr=yerr, fmt=".", markersize=4, color=color, alpha=0.7, label=f"{label} simulation")
plt.fill_between(x_fit, np.maximum(y_fit - 1.96 * s, 0), y_fit + 1.96 * s, color="lightblue", alpha=0.3, label="95% scatter interval")
plt.fill_between(x_fit, np.maximum(y_fit - s, 0), y_fit + s, color="pink", alpha=0.35, label="68% scatter interval")
plt.plot(x_fit, y_fit, color="black", linewidth=2, label="Rational curve fit")
equation = (
    r"$\frac{h_{\mathrm{rim}}}{R_{\mathrm{meas}}}=\frac{a}{K/R^2+b}$" "\n"
    rf"$a={a:.6f}\pm{a_err:.6f}$" "\n"
    rf"$b={b:.6f}\pm{b_err:.6f}$"
)
plt.text(0.97, 0.97, equation, transform=plt.gca().transAxes, fontsize=11,
         ha="right", va="top", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
plt.xlabel(r"Degradation state $K/R^2$")
plt.ylabel(r"Rim height $h_{\mathrm{rim}}/R_{\mathrm{meas}}$")
plt.title("Rim height vs. degradation state — complexity 5")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.close()

order = np.argsort(yhat)
plt.figure(figsize=(7, 6))
plt.scatter(yhat, residuals ** 2, s=10, alpha=0.3, label="Squared residuals")
plt.plot(yhat[order], (np.exp(c) * yhat ** alpha)[order], "k-", linewidth=2,
         label=rf"$\sigma_y^2=e^{{{c:.3f}}}\hat{{y}}^{{{alpha:.3f}}}$")
plt.xlabel("Predicted normalized rim height")
plt.ylabel("Squared residual")
plt.title("Variance model — complexity 5")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(variance_png, dpi=300)
plt.close()

print(f"a = {a:.6f} +/- {a_err:.6f}")
print(f"b = {b:.6f} +/- {b_err:.6f}")
print(f"alpha = {alpha:.6f}\nc = {c:.6f}")
print(f"Initial fitted rim height = {a / b:.6f}")
print(f"Saved {calibration_csv}, {output_png}, {variance_png}")
