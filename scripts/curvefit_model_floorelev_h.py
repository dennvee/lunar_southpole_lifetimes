from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
PLOT_DIR = ROOT / 'plots'
nominal_radius = 500.0
files = [f'morphometry_vs_K_rim_{h}m.csv' for h in (35, 40, 50, 60)]
calibration_csv = DATA_DIR / f'floor_calibration.csv'
plot_png = PLOT_DIR / f'floor_calibration.png'
variance_png = PLOT_DIR / f'floor_variance.png'

def func(x, y0, a, p):
    return y0 * (a / (x + a)) ** p

def sigma_model(y, alpha, c):
    return np.exp(c / 2) * y ** (alpha / 2)

datasets = []
for h, filename in zip((35, 40, 50, 60), files):
    df = pd.read_csv(DATA_DIR / filename)
    x = df['K_over_R2'].to_numpy(dtype=float)
    radius = df['radius'].to_numpy(dtype=float)
    y = np.abs(df['floor_elevation_over_R'].to_numpy(dtype=float))
    yerr = df['floor_elevation_std'].to_numpy(dtype=float) / radius
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0) & (radius > 0)
    datasets.append((x[valid], y[valid], yerr[valid], f'{h} m'))

x_all = np.concatenate([d[0] for d in datasets])
y_all = np.concatenate([d[1] for d in datasets])
p0 = [np.mean([d[1][0] for d in datasets]), 0.1, 3.0]
bounds = ([1e-8, 1e-8, 1e-8], [np.inf, np.inf, np.inf])
popt, pcov = curve_fit(func, x_all, y_all, p0=p0, bounds=bounds, maxfev=30000)
for iteration in range(10):
    yhat = func(x_all, *popt)
    residuals = y_all - yhat
    valid = (yhat > 0) & np.isfinite(residuals) & (np.abs(residuals) > 1e-12)
    alpha, c = np.polyfit(np.log(yhat[valid]), np.log(residuals[valid] ** 2), 1)
    sigma = np.maximum(sigma_model(yhat, alpha, c), 1e-10)
    updated, covariance = curve_fit(func, x_all, y_all, p0=popt, bounds=bounds,
                                    sigma=sigma, absolute_sigma=True, maxfev=30000)
    converged = np.allclose(updated, popt, rtol=1e-6, atol=1e-9)
    popt, pcov = updated, covariance
    if converged:
        break

yhat = func(x_all, *popt)
residuals = y_all - yhat
valid = (yhat > 0) & (np.abs(residuals) > 1e-12)
alpha, c = np.polyfit(np.log(yhat[valid]), np.log(residuals[valid] ** 2), 1)
y0, a, p = popt
y0_err, a_err, p_err = np.sqrt(np.diag(pcov))
pd.DataFrame([dict(y0=y0, a=a, p=p, y0_err=y0_err, a_err=a_err,
                   p_err=p_err, alpha=alpha, c=c, x_max=x_all.max())]).to_csv(calibration_csv, index=False)

x_fit = np.linspace(0, x_all.max(), 1000)
y_fit = func(x_fit, *popt)
s = sigma_model(y_fit, alpha, c)
plt.figure(figsize=(8, 6))
for x, y, yerr, label in datasets:
    plt.errorbar(x, y, yerr=yerr, fmt='.', markersize=3, alpha=0.6, label=f'{label} simulation')
plt.fill_between(x_fit, np.maximum(y_fit - 1.96*s, 0), y_fit + 1.96*s, alpha=0.15, label='95% interval')
plt.fill_between(x_fit, np.maximum(y_fit - s, 0), y_fit + s, alpha=0.25, label='68% interval')
plt.plot(x_fit, y_fit, 'k-', lw=2, label='Heteroskedastic fit')
plt.xlabel(r'Degradation state $K/R^2$')
plt.ylabel(r'Floor depth $|h_{floor}|/R_{meas}$')
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plot_png, dpi=300)
plt.close()

order = np.argsort(yhat)
plt.figure(figsize=(7, 6))
plt.scatter(yhat, residuals**2, s=10, alpha=0.3, label='Squared residuals')
plt.plot(yhat[order], (np.exp(c)*yhat**alpha)[order], 'k-', lw=2, label='Predicted variance')
plt.xlabel('Predicted normalized floor depth')
plt.ylabel('Squared residual / predicted variance')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(variance_png, dpi=300)
plt.close()
print(f'y0 = {y0:.6f} +/- {y0_err:.6f}\na = {a:.6f} +/- {a_err:.6f}\np = {p:.6f} +/- {p_err:.6f}')
print(f'alpha = {alpha:.6f}\nc = {c:.6f}')
print(f'Saved {calibration_csv}, {plot_png}, {variance_png}')