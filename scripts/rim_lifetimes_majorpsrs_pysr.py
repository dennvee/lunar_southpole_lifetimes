from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

input_csv = DATA_DIR / "real_rim_K_majorpsrs_c5_psrs_002.csv"
kappa_csv = DATA_DIR / "kappa_trask_flat2.csv"
output_csv = DATA_DIR / "rim_lifetimes_majorpsrs_c5_psrs_002.csv"

real = pd.read_csv(input_csv)
kap = pd.read_csv(kappa_csv, header=None, usecols=[0, 1], names=["D_km", "kappa"])
kap = kap.dropna().sort_values("D_km")

if (kap[["D_km", "kappa"]] <= 0).any().any() or kap["D_km"].duplicated().any():
    raise ValueError("Kappa table must have positive values and unique diameters.")

R = real["fitted_radius_m"].to_numpy(dtype=float)
if not np.all(np.isfinite(R) & (R > 0)):
    raise ValueError("All fitted crater radii must be positive and finite.")

D_km = 2 * R / 1000
log_kappa = interp1d(
    np.log10(kap["D_km"]),
    np.log10(kap["kappa"]),
    kind="linear",
    fill_value="extrapolate",
)
kappa = 10 ** log_kappa(np.log10(D_km))

outside = (D_km < kap["D_km"].min()) | (D_km > kap["D_km"].max())
if outside.any():
    print(f"Warning: extrapolating kappa for {outside.sum()} of {len(real)} craters.")

real["diameter_km"] = D_km
real["kappa_m2_per_Myr"] = kappa
real["kappa_extrapolated"] = outside

for label in ["best", "68_low", "68_high", "95_low", "95_high"]:
    x = real[f"K_over_R2_{label}"].to_numpy(dtype=float)
    real[f"t_{label}_Myr"] = x * R**2 / kappa

real.to_csv(output_csv, index=False)
print(f"Valid best-fit lifetimes: {real['t_best_Myr'].notna().sum()} of {len(real)}")
print(f"Saved {output_csv}")
