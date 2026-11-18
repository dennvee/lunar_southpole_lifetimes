from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from pysr import PySRRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

heights = [35, 40, 50, 60]
bin_width = 0.001
min_fit_x = 0.0
selected_complexities = [5, 6, 8, 9]
datasets = []

for h in heights:
    df = pd.read_csv(DATA_DIR / f"morphometry_vs_K_rim_{h}m_renormalized_004.csv")
    df = df[["K_over_R2", "floor_elevation_over_R"]].dropna()
    df["floor_depth_over_R"] = df["floor_elevation_over_R"].abs()
    df = df[np.isfinite(df["K_over_R2"]) & np.isfinite(df["floor_depth_over_R"]) & (df["K_over_R2"] >= 0) & (df["floor_depth_over_R"] > 0)]
    df["initial_rim_height"] = h
    datasets.append(df)

combined = pd.concat(datasets, ignore_index=True)

combined["bin"] = np.where(
    combined["K_over_R2"] == 0,
    -1,
    np.floor(combined["K_over_R2"] / bin_width).astype(int))

per_simulation = combined.groupby(
    ["bin", "initial_rim_height"], as_index=False
).agg(
    K_over_R2=("K_over_R2", "mean"),
    floor_depth_over_R=("floor_depth_over_R", "mean"))

df_mean = per_simulation.groupby("bin", as_index=False).agg(
    K_over_R2=("K_over_R2", "mean"),
    floor_depth_mean=("floor_depth_over_R", "mean"),
    floor_depth_std=("floor_depth_over_R", "std"),
    n=("floor_depth_over_R", "count"))

df_mean = df_mean.sort_values("K_over_R2")

if len(df_mean) < 2:
    raise ValueError("Not enough degradation bins to fit PySR.")

df_mean.to_csv(DATA_DIR / "pysr_floorelev_mean_allbins_004.csv", index=False)
x = df_mean["K_over_R2"].to_numpy()
y = df_mean["floor_depth_mean"].to_numpy()

fit_mask = x >= min_fit_x
if fit_mask.sum() < 2:
    raise ValueError("Not enough points in the selected fitting range.")

X_fit = x[fit_mask].reshape(-1, 1)
y_fit = y[fit_mask]

model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp"],
    random_state=42,
    deterministic=True,
    parallelism="serial")

model.fit(X_fit, y_fit, variable_names=["x"])
eq = model.equations_

selected = [
    eq.loc[eq["complexity"] == c, "loss"].idxmin()
    for c in selected_complexities
    if (eq["complexity"] == c).any()]

if not selected:
    raise ValueError("None of the selected complexities were found.")

x_line = np.linspace(x.min(), x.max(), 500)
X_line = x_line.reshape(-1, 1)
colors = ["#f8bbd0", "#f48fb1", "#ec407a", "#ad1457"]

fig, (ax, ax_eq) = plt.subplots(
    1, 2, figsize=(17, 7),
    gridspec_kw={"width_ratios": [1.4, 1]})

full = df_mean["n"] == len(heights)
ax.scatter(x[full], y[full], color="navy", s=25, alpha=0.7,
           label="Mean (4 simulations)", zorder=3)
ax.scatter(x[~full], y[~full], color="royalblue", marker="o", s=30,
           facecolors="none", label="Mean (1–3 simulations)", zorder=3)

ax_eq.axis("off")

for i, idx in enumerate(selected):
    color = colors[i]
    complexity = int(eq.loc[idx, "complexity"])

    ax.plot(
        x_line, model.predict(X_line, index=idx),
        color=color, lw=2.5,
        label=f"Complexity {complexity}")

    expr = model.sympy(index=idx)
    constants = list(dict.fromkeys(
        node for node in sp.preorder_traversal(expr)
        if isinstance(node, sp.Float)))

    names = ["a", "p", "b", "c", "d", "e", "f", "g", "j", "k"]
    parameters = [sp.Symbol(name) for name in names[:len(constants)]]

    symbolic = expr.xreplace(dict(zip(constants, parameters)))
    values = ", ".join(
        rf"{sp.latex(name)}={float(value):.4g}"
        for name, value in zip(parameters, constants))

    y_pos = 0.95 - i * 0.23

    ax_eq.text(
        0.02, y_pos, f"Complexity {complexity}",
        color=color, fontsize=12, fontweight="bold",
        transform=ax_eq.transAxes, va="top")

    ax_eq.text(
        0.02, y_pos - 0.06,
        rf"$y={sp.latex(symbolic)}$",
        fontsize=12, transform=ax_eq.transAxes, va="top" )

    if values:
        ax_eq.text(
            0.02, y_pos - 0.12,
            rf"${values}$",
            fontsize=10, transform=ax_eq.transAxes, va="top")
ax.set_xlabel(r"$K/R^2$")
ax.set_ylabel(r"$|h_{\mathrm{floor}}|/R$")
ax.set_title("PySR Floor Depth Calibration — Mean Simulations")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(
    PLOT_DIR / "pysr_floorelev_calibration_mean_allbins_004.png",
    dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Number of averaged degradation states: {len(df_mean)}")
print(f"Simulation contributions per bin:\n{df_mean['n'].value_counts().sort_index().to_string()}")
print(f"Number of points used for fitting: {len(y_fit)}")
print(f"Fitting range: {x[fit_mask].min():.6f} to {x[fit_mask].max():.6f}")
print(eq[["complexity", "loss", "score", "equation"]])