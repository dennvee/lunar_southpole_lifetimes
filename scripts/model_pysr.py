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

df = pd.read_csv(DATA_DIR / "morphometry_vs_K_rim_35m_renormalized_003.csv")
x = df["K_over_R2"].to_numpy()
y = df["rim_height_over_R"].to_numpy()
X = x.reshape(-1, 1)

model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp"],
    random_state=42,
    deterministic=True,
    parallelism="serial",
)

model.fit(X, y, variable_names=["x"])

eq = model.equations_
simple_exp = eq[(eq["complexity"] <= 10) & eq["equation"].str.contains("exp", regex=False)]
simple = eq[eq["complexity"] <= 10]

selected = []
for subset in (simple_exp, simple):
    if not subset.empty:
        selected.append(subset["loss"].idxmin())

selected += [model.get_best().name, eq["loss"].idxmin()]
selected = list(dict.fromkeys(selected))

x_line = np.linspace(x.min(), x.max(), 500)
X_line = x_line.reshape(-1, 1)
colors = ["#f8bbd0", "#f48fb1", "#ec407a", "#ad1457"]

fig, ax = plt.subplots(figsize=(11, 7))
ax.scatter(x, y, color="navy", s=25, alpha=0.7,
           label="Simulation data", zorder=3)

equations = []
for idx, color in zip(selected, colors):
    complexity = int(eq.loc[idx, "complexity"])

    ax.plot(x_line, model.predict(X_line, index=idx),
            color=color, lw=2.5, label=f"Complexity {complexity}")

    expr = model.sympy(index=idx)
    constants = list(dict.fromkeys(
        node for node in sp.preorder_traversal(expr)
        if isinstance(node, sp.Float)
    ))

    names = ["a", "p", "b", "c", "d", "e", "f", "g", "j", "k"]
    parameters = [sp.Symbol(name) for name in names[:len(constants)]]

    symbolic = expr.xreplace(dict(zip(constants, parameters)))
    values = ", ".join(
        rf"{name}={float(value):.4g}"
        for name, value in zip(parameters, constants)
    )

    equations.append(
        rf"$C={complexity}:\quad y={sp.latex(symbolic)}$"
        + "\n" + rf"${values}$"
    )

ax.set_xlabel(r"$K/R^2$")
ax.set_ylabel(r"$h_{\mathrm{rim}}/R$")
ax.set_title("PySR Rim Height Calibration")
ax.legend()
ax.grid(alpha=0.25)

fig.text(0.5, -0.03, "\n\n".join(equations),
         ha="center", va="top", fontsize=9)

fig.savefig(PLOT_DIR / "pysr_rim_calibration.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
print(eq[["complexity", "loss", "score", "equation"]])