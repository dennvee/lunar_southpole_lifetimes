from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'morphometryv2'
RADIUS = 500.0
FILES = [f'morphometry_vs_K_rim_{h}m_d1000m_renormalized.csv' for h in (35, 40, 50, 60)]
CALIBRATION_CSV = f'floor_radius_calibration_R{RADIUS:g}m.csv'
PLOT_PNG = f'floor_radius_heteroskedastic_R{RADIUS:g}m.png'
VARIANCE_PNG = f'floor_radius_variance_R{RADIUS:g}m.png'

def func(x, y0, tau, m):
    return y0 * (np.exp(-x / tau) + m * x)

def estimate_variance(x, y, parameters):
    prediction = func(x, *parameters)
    residuals = y - prediction
    valid = np.isfinite(prediction) & (prediction > 0) & np.isfinite(residuals) & (np.abs(residuals) > 1e-12)
    if valid.sum() < 3 or np.ptp(np.log(prediction[valid])) < 1e-9:
        raise ValueError('Insufficient variation for heteroskedastic variance regression')
    alpha, c = np.polyfit(np.log(prediction[valid]), np.log(residuals[valid] ** 2), 1)
    return alpha, c

datasets = []
for file in FILES:
    df = pd.read_csv(DATA_DIR / file)
    radius = df['radius'].to_numpy(dtype=float)
    x = df['K_over_R2'].to_numpy(dtype=float)
    y = df['floor_radius'].to_numpy(dtype=float) / radius
    yerr = df['floor_radius_std'].to_numpy(dtype=float) / radius
    valid = np.isfinite(x) & np.isfinite(y) & (radius > 0) & (y > 0.0)
    label = file.split('_rim_')[1].split('m')[0] + ' m'
    datasets.append((x[valid], y[valid], yerr[valid], label))

x_all = np.concatenate([d[0] for d in datasets])
y_all = np.concatenate([d[1] for d in datasets])
y0_guess = np.mean([d[1][0] for d in datasets])
p0 = [y0_guess, 0.02, 5.0]
bounds = ([1e-8, 1e-6, 0.0], [np.inf, np.inf, np.inf])
popt, pcov = curve_fit(func, x_all, y_all, p0=p0, bounds=bounds, maxfev=50000)

for _ in range(10):
    alpha, c = estimate_variance(x_all, y_all, popt)
    sigma = np.exp(c / 2) * np.maximum(func(x_all, *popt), 1e-12) ** (alpha / 2)
    sigma = np.maximum(sigma, 1e-8)
    updated, updated_cov = curve_fit(func, x_all, y_all, p0=popt, bounds=bounds,
                                     sigma=sigma, absolute_sigma=True, maxfev=50000)
    converged = np.allclose(updated, popt, rtol=1e-6, atol=1e-9)
    popt, pcov = updated, updated_cov
    if converged:
        break

alpha, c = estimate_variance(x_all, y_all, popt)
y0, tau, m = popt
errors = np.sqrt(np.diag(pcov))
x_max = float(x_all.max())
minimum = minimize_scalar(lambda x: func(x, *popt), bounds=(0.0, x_max), method='bounded')
x_turn = float(minimum.x)
if not (0 < x_turn < x_max) or not (func(x_turn, *popt) < min(func(0, *popt), func(x_max, *popt))):
    raise ValueError('The fitted mean curve has no interior minimum in the calibration range')

pd.DataFrame([dict(y0=y0, tau=tau, m=m, y0_err=errors[0], tau_err=errors[1],
                   m_err=errors[2], alpha=alpha, c=c, K_max=x_max, K_turn=x_turn)]).to_csv(CALIBRATION_CSV, index=False)

x_fit = np.linspace(0, x_max, 1200)
y_fit = func(x_fit, *popt)
sigma_fit = np.exp(c / 2) * y_fit ** (alpha / 2)
plt.figure(figsize=(9, 6))
for x, y, yerr, label in datasets:
    plt.errorbar(x, y, yerr=yerr, fmt='.', alpha=0.4, markersize=3,
                 capsize=1, label=f'{label} simulation')
plt.fill_between(x_fit, np.maximum(y_fit - 1.96*sigma_fit, 0), y_fit + 1.96*sigma_fit, alpha=0.15, label='95% band')
plt.fill_between(x_fit, np.maximum(y_fit - sigma_fit, 0), y_fit + sigma_fit, alpha=0.25, label='68% band')
plt.plot(x_fit, y_fit, 'k-', linewidth=2, label='Heteroskedastic mean fit')
plt.axvline(x_turn, color='gray', linestyle=':', label=f'Minimum at K/R²={x_turn:.4f}')
plt.xlabel(r'Degradation state $K/R^2$')
plt.ylabel(r'Floor radius $r_{\rm floor}/R_{\rm measured}$')
plt.title('Floor radius vs. degradation state')
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_PNG, dpi=300)
plt.close()

pred = func(x_all, *popt)
plt.figure(figsize=(8, 6))
plt.scatter(pred, (y_all - pred)**2, s=9, alpha=0.3, label='Squared residuals')
order = np.argsort(pred)
plt.plot(pred[order], np.exp(c)*pred[order]**alpha, color='black', label='Variance model')
plt.xlabel('Predicted normalized floor radius')
plt.ylabel('Squared residual / fitted variance')
plt.title('Floor radius heteroskedastic variance')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(VARIANCE_PNG, dpi=300)
plt.close()
print(f'y0={y0:.6f} +/- {errors[0]:.6f}; tau={tau:.6f} +/- {errors[1]:.6f}; m={m:.6f} +/- {errors[2]:.6f}')
print(f'alpha={alpha:.6f}; c={c:.6f}; turning point K/R²={x_turn:.6f}')
print('Saved:', CALIBRATION_CSV, PLOT_PNG, VARIANCE_PNG)