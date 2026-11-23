from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, PLOT_DIR = ROOT / "data", ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

def c6(x, A, a):
    return A * np.exp(-a * x)

def c8(x, A, a, b):
    return A * np.exp(-a * x) + b

df = pd.read_csv(DATA_DIR / "pysr_floorelev_mean_fixedR_004.csv")
x, y = (df[col].to_numpy(dtype=float) for col in ("K_over_R2", "floor_depth_mean"))
if len(x) < 4 or not np.isfinite(x).all() or not np.isfinite(y).all() or (x < 0).any() or (y <= 0).any():
    raise ValueError("Expected at least four finite, positive floor depths and nonnegative degradation states.")

models = {
    6: (c6, [0.32788604, 8.796599], "#e76f9b"),
    8: (c8, [0.31907374, 10.440555, 0.018970076], "#a31960"),}
results, rows = {}, []
for complexity, (func, p0, color) in models.items():
    popt, pcov = curve_fit(func, x, y, p0=p0, bounds=(0, np.inf), maxfev=30000)
    errors = np.sqrt(np.diag(pcov))
    yhat = func(x, *popt)
    residuals = y - yhat
    mask = (np.abs(residuals) > 1e-12) & (yhat > 0)
    if mask.sum() < 3:
        raise ValueError(f"Not enough nonzero residuals for complexity {complexity}.")
    alpha, c = np.polyfit(np.log(yhat[mask]), np.log(residuals[mask] ** 2), 1)
    mse = np.mean(residuals**2)
    results[complexity] = (func, popt, errors, yhat, residuals, alpha, c, color)
    rows.append(dict(complexity=complexity, A=popt[0], a=popt[1],
                     b=popt[2] if complexity == 8 else 0.0,
                     A_err=errors[0], a_err=errors[1],
                     b_err=errors[2] if complexity == 8 else np.nan,
                     alpha=alpha, c=c, MSE=mse, RMSE=np.sqrt(mse),
                     x_min=x.min(), x_max=x.max(), n_states=len(x)))

pd.DataFrame(rows).to_csv(DATA_DIR / "floor_calibration_compare_c6_c8_fixedR_004.csv", index=False)
x_line = np.linspace(x.min(), x.max(), 1000)
xlabel, ylabel = r"$K/R_0^2$ ($R_0=500$ m)", r"$|h_{\mathrm{floor}}|/R_0$"

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(x, y, s=13, color="navy", alpha=0.6, label="Mean of 4 simulations")
for complexity, (func, popt, _, _, residuals, _, _, color) in results.items():
    ax.plot(x_line, func(x_line, *popt), color=color, lw=2.2,
            label=f"Complexity {complexity} (RMSE={np.sqrt(np.mean(residuals**2)):.5f})")
ax.set(xlabel=xlabel, ylabel=ylabel, title="Fixed-radius floor-depth calibration: complexity 6 vs 8")
ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
fig.savefig(PLOT_DIR / "floor_calibration_compare_c6_c8_fixedR_004.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
for complexity, (_, _, _, _, residuals, _, _, color) in results.items():
    ax.plot(x, residuals, color=color, lw=1.6, label=f"Complexity {complexity}")
ax.axhline(0, color="black", lw=1, ls="--")
ax.set(xlabel=r"$K/R_0^2$", ylabel="Observed mean − fitted mean", title="Residual comparison: complexity 6 vs 8")
ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
fig.savefig(PLOT_DIR / "floor_residuals_compare_c6_c8_fixedR_004.png", dpi=300)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (complexity, (_, _, _, yhat, residuals, alpha, c, color)) in zip(axes, results.items()):
    order = np.argsort(yhat)
    ax.scatter(yhat, residuals**2, s=13, alpha=0.4, color=color, label="Squared residual")
    ax.plot(yhat[order], np.exp(c) * yhat[order]**alpha, color="black", lw=2, label="Log-residual model")
    ax.set(xlabel="Predicted normalized floor depth", ylabel="Squared mean-curve residual",
           title=f"Complexity {complexity}: α={alpha:.3f}, c={c:.3f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(PLOT_DIR / "floor_variance_compare_c6_c8_fixedR_004.png", dpi=300)
plt.close(fig)

for complexity, (func, popt, errors, _, _, alpha, c, _) in results.items():
    fit = func(x_line, *popt)
    sigma = np.sqrt(np.exp(c) * fit**alpha)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, color="navy", s=15, alpha=0.65, label="Mean of 4 simulations")
    for multiplier, color, label in [(1.96, "lightblue", "1.96σ residual band"), (1, "lightpink", "1σ residual band")]:
        ax.fill_between(x_line, np.maximum(fit - multiplier*sigma, 0), fit + multiplier*sigma,
                        color=color, alpha=0.5 if multiplier == 1.96 else 0.6, label=label)
    ax.plot(x_line, fit, color="black", lw=2, label=f"Complexity {complexity} fit")
    equation = r"$y=Ae^{-ax}$" if complexity == 6 else r"$y=Ae^{-ax}+b$"
    params = "\n".join(f"{name}={val:.6f} ± {err:.6f}"
                       for name, val, err in zip("Aab", popt, errors))
    ax.text(0.97, 0.97, equation + "\n" + params, transform=ax.transAxes,
            va="top", ha="right", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    ax.set(xlabel=xlabel, ylabel=ylabel, title=f"Fixed-radius floor-depth calibration: complexity {complexity}")
    ax.legend(loc="lower left"); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(PLOT_DIR / f"floor_calibration_c{complexity}_fixedR_004.png", dpi=300)
    plt.close(fig)

for row in rows:
    print(f"Complexity {row['complexity']}: A={row['A']:.7f} ± {row['A_err']:.7f}, a={row['a']:.7f} ± {row['a_err']:.7f}, b={row['b']:.7f}, RMSE={row['RMSE']:.7f}, alpha={row['alpha']:.4f}, c={row['c']:.4f}")
print("Saved calibration CSV and comparison, residual, variance, and individual calibration plots.")