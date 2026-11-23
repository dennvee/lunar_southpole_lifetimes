from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from pysr import PySRRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, PLOT_DIR = ROOT / "data", ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

heights = [35, 40, 50, 60]
R0 = 500.0
complexities = [5, 6, 8, 9]
datasets = [pd.read_csv(DATA_DIR / f"morphometry_vs_K_rim_{h}m_renormalized_004.csv") for h in heights]
K = datasets[0]["K"].to_numpy()
depths = -np.column_stack([df["floor_elevation"].to_numpy() for df in datasets]) / R0

x, y = K / R0**2, depths.mean(axis=1)
pd.DataFrame({"K": K, "K_over_R2": x, "floor_depth_mean": y,
              "floor_depth_std": depths.std(axis=1, ddof=1)}).to_csv(
    DATA_DIR / "pysr_floorelev_mean_fixedR_004.csv", index=False)

model = PySRRegressor(
    niterations=100, binary_operators=["+", "-", "*", "/"], unary_operators=["exp"],
    random_state=42, deterministic=True, parallelism="serial")
model.fit(x[:, None], y, variable_names=["x"])
eq = model.equations_
selected = [eq.loc[eq["complexity"] == c, "loss"].idxmin()
            for c in complexities if (eq["complexity"] == c).any()]

xx = np.linspace(x.min(), x.max(), 500)
colors = ["#f8bbd0", "#f48fb1", "#ec407a", "#ad1457"]
fig, (ax, ax_eq) = plt.subplots(1, 2, figsize=(17, 7),
                               gridspec_kw={"width_ratios": [1.4, 1]})
ax.scatter(x, y, color="navy", s=25, alpha=0.7,
           label=f"Mean ({len(heights)} simulations)", zorder=3)
ax_eq.axis("off")

for i, idx in enumerate(selected):
    c, color = int(eq.loc[idx, "complexity"]), colors[i]
    ax.plot(xx, model.predict(xx[:, None], index=idx), color=color,
            lw=2.5, label=f"Complexity {c}")
    expr = model.sympy(index=idx)
    constants = list(dict.fromkeys(n for n in sp.preorder_traversal(expr) if isinstance(n, sp.Float)))
    symbols = sp.symbols("a p b c d e f g j k")[:len(constants)]
    symbolic = expr.xreplace(dict(zip(constants, symbols)))
    values = ", ".join(rf"{sp.latex(s)}={float(v):.4g}" for s, v in zip(symbols, constants))
    pos = 0.95 - i * 0.23
    ax_eq.text(0.02, pos, f"Complexity {c}", color=color, fontsize=12,
               fontweight="bold", transform=ax_eq.transAxes, va="top")
    ax_eq.text(0.02, pos - 0.06, rf"$y={sp.latex(symbolic)}$" +
               ("\n" + rf"${values}$" if values else ""),
               fontsize=11, transform=ax_eq.transAxes, va="top", linespacing=1.8)

ax.set(xlabel=r"$K/R_0^2$ ($R_0=500\ \mathrm{m}$)",
       ylabel=r"$|h_{\mathrm{floor}}|/R_0$",
       title="Floor Depth Calibration")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(PLOT_DIR / "floorelev_calibrationrerun_data004.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Number of averaged degradation states: {len(x)}")
print(f"Simulations per state: {len(heights)}")
print(f"Fitting range: {x.min():.6f} to {x.max():.6f}")
print(eq[["complexity", "loss", "score", "equation"]])