from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[1]
nominal_radius = 500.0

input_csv = ROOT / "scripts" / f"real_rim_K_small_R{nominal_radius:g}m.csv"
kappa_csv = ROOT / "trask" / "kappa_trask_flat2.csv"
output_csv = ROOT / f"rim_small_lifetimes_R{nominal_radius:g}m.csv"

real = pd.read_csv(input_csv)
kap = pd.read_csv(kappa_csv, header=None, usecols=[0, 1], names=["D_km", "kappa"])
kap = kap.sort_values("D_km")

R = real["fitted_radius_m"].to_numpy(dtype=float)
D_km = 2 * R / 1000

log_kappa = interp1d(
    np.log10(kap["D_km"]), np.log10(kap["kappa"]),
    kind="linear", fill_value="extrapolate")

kappa = 10 ** log_kappa(np.log10(D_km))
real["diameter_km"] = D_km
real["kappa_m2_per_Myr"] = kappa
for label in ["best", "68_low", "68_high", "95_low", "95_high"]:
    real[f"t_{label}_Myr"] = real[f"K_over_R2_{label}"] * R**2 / kappa
real.to_csv(output_csv, index=False)