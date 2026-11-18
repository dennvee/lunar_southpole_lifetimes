from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

input_csv = DATA_DIR / "real_floor_K_majorpsrs_c6_mean_004.csv"
kappa_csv = DATA_DIR / "kappa_trask_flat2.csv"
output_csv = DATA_DIR / "floor_lifetimes_majorpsrs_c6_mean_004.csv"

real = pd.read_csv(input_csv)
kap = pd.read_csv(kappa_csv, header=None, usecols=[0, 1], names=["D_km", "kappa"])

kap = kap.sort_values("D_km")
if not np.all(np.isfinite(kap[["D_km", "kappa"]])) or (kap[["D_km", "kappa"]] <= 0).any().any():
    raise ValueError("Trask table must contain finite, positive diameters and diffusivities.")
if kap["D_km"].duplicated().any() or len(kap) < 2:
    raise ValueError("Trask table needs at least two unique crater diameters.")

R = real["fitted_radius_m"].to_numpy(dtype=float)
if np.any(~np.isfinite(R) | (R <= 0)):
    raise ValueError("Real-crater fitted radii must be finite and positive.")

D_km = 2 * R / 1000
log_kappa = interp1d(
    np.log10(kap["D_km"]),
    np.log10(kap["kappa"]),
    kind="linear",
    fill_value="extrapolate",
)
kappa = 10 ** log_kappa(np.log10(D_km))

real["diameter_km"] = D_km
real["kappa_m2_per_Myr"] = kappa
real["kappa_extrapolated"] = (D_km < kap["D_km"].min()) | (D_km > kap["D_km"].max())

for label in ["best", "68_low", "68_high", "95_low", "95_high"]:
    real[f"t_{label}_Myr"] = real[f"K_over_R2_{label}"] * R**2 / kappa

real.to_csv(output_csv, index=False)

for _, row in real.iterrows():
    name = row.get("PSR_NAME", row.get("CRATER_ID", "Crater"))
    print(
        f"{name}: {row['t_best_Myr']:.3f} Myr "
        f"(68% {row['t_68_low_Myr']:.3f}–{row['t_68_high_Myr']:.3f}; "
        f"95% {row['t_95_low_Myr']:.3f}–{row['t_95_high_Myr']:.3f})"
    )

print(f"Trask diffusivity extrapolated for {real['kappa_extrapolated'].sum()} of {len(real)} craters")
print(f"Saved {output_csv}")