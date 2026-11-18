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

mean_csv = DATA_DIR / "pysr_floorelev_mean_allbins_004.csv"
calibration_csv = DATA_DIR / "floor_calibration_complexity6_mean_004.csv"
plot_png = PLOT_DIR / "curve_fit_floor_complexity6_mean_004.png"
variance_png = PLOT_DIR / "floor_variance_complexity6_mean_004.png"

def func(x, A, a):
    return A * np.exp(-a * x)

def sigma_model(y, alpha, c):
    return np.exp(c / 2) * y ** (alpha / 2)

df = pd.read_csv(mean_csv)
x = df["K_over_R2"].to_numpy(dtype=float)
y = df["floor_depth_mean"].to_numpy(dtype=float)
valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y > 0)
x, y = x[valid], y[valid]
if len(x) < 3:
    raise ValueError("Not enough valid binned means to fit.")

popt, pcov = curve_fit(
    func, x, y,
    p0=[0.34629795, 13.822333],
    bounds=([1e-12, 1e-12], [np.inf, np.inf]),
    maxfev=30000)
A, a = popt
A_err, a_err = np.sqrt(np.diag(pcov))
yhat = func(x, *popt)
residuals = y - yhat
mask = np.isfinite(residuals) & (np.abs(residuals) > 1e-12)
if mask.sum() < 3:
    raise ValueError("Insufficient nonzero residuals for the variance model.")
alpha, c = np.polyfit(np.log(yhat[mask]), np.log(residuals[mask] ** 2), 1)

pd.DataFrame([{
    "A": A, "a": a, "A_err": A_err, "a_err": a_err,
    "alpha": alpha, "c": c, "x_min": x.min(), "x_max": x.max(),
    "n_bins": len(x),}]).to_csv(calibration_csv, index=False)

x_fit = np.linspace(x.min(), x.max(), 1000)
y_fit = func(x_fit, *popt)
s = sigma_model(y_fit, alpha, c)
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color="navy", s=20, alpha=0.7, label="Binned simulation means")
plt.fill_between(x_fit, np.maximum(y_fit - 1.96 * s, 0), y_fit + 1.96 * s,
                 color="lightblue", alpha=0.4, label="1.96σ scatter band")
plt.fill_between(x_fit, np.maximum(y_fit - s, 0), y_fit + s,
                 color="lightpink", alpha=0.5, label="1σ scatter band")
plt.plot(x_fit, y_fit, "k-", lw=2, label="Complexity-6 fit")
label = (
    r"$|h_{\mathrm{floor}}|/R=Ae^{-a(K/R^2)}$" "\n"
    rf"$A={A:.6f}\pm{A_err:.6f}$" "\n"
    rf"$a={a:.6f}\pm{a_err:.6f}$")
plt.text(0.97, 0.97, label, transform=plt.gca().transAxes,
         ha="right", va="top", fontsize=11,
         bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
plt.xlabel(r"Degradation state $K/R^2$")
plt.ylabel(r"Floor depth $|h_{\mathrm{floor}}|/R$")
plt.title("Floor-depth binned-mean calibration — complexity 6")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plot_png, dpi=300)
plt.close()

order = np.argsort(yhat)
plt.figure(figsize=(7, 6))
plt.scatter(yhat, residuals ** 2, s=15, alpha=0.4, label="Squared residuals")
plt.plot(yhat[order], (np.exp(c) * yhat ** alpha)[order], "k-", lw=2,
         label="Predicted residual variance")
plt.xlabel("Predicted normalized floor depth")
plt.ylabel("Squared residual")
plt.title("Binned-mean residual variance — complexity 6")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(variance_png, dpi=300)
plt.close()
print(f"Fitted {len(x)} binned means over {x.min():.6f} to {x.max():.6f}")
print(f"A = {A:.6f} +/- {A_err:.6f}")
print(f"a = {a:.6f} +/- {a_err:.6f}")
print(f"alpha = {alpha:.6f}; c = {c:.6f}")
print(f"Saved {calibration_csv}, {plot_png}, {variance_png}")