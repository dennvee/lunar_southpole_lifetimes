from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cratermaker as cm
from morphometry import get_morphometry_from_binned_profile as mbp
from morphometry import save_morphometric_parameters

ROOT = Path(__file__).resolve().parents[1]
rim_height_initial = float(sys.argv[1])
output_csv = ROOT / "data" / f"morphometry_vs_K_rim_{int(rim_height_initial)}m_002.csv"

diameter = 1000
radius = diameter / 2
location = (0.0, 0.0)
K_increment = 250
K_values = np.arange(0, 60000, K_increment)

sim = cm.Simulation(
    simdir=ROOT / "simulation_directories" / f"sim_rim_{int(rim_height_initial)}m_002",
    surface="hireslocal",
    morphology="basicmoon",
    local_location=location, local_radius=2 * diameter,
    pix=4.0, reset=True,
    conserve_volume=False, ask_overwrite=False)

crater = sim.Crater.maker(
    diameter=diameter,
    location=location,
    name=f"crater_rim_{int(rim_height_initial)}m",
    rim_height=rim_height_initial)

sim.emplace(crater)

fit_vars = [
    "radius",
    "rim_height", "rim_width",
    "floor_radius", "wall_curvature",
    "floor_blend", "frac_ejrim", "ejprofile"]

for i, K in enumerate(K_values):

    if i == 0:
        print(crater)

    if i > 0:
        sim.surface.apply_diffusion(K_increment)

    fit, uncertainty = mbp(
        sim=sim, crater=crater,
        fit_vars=fit_vars, nrbins=100,
        fit_range=(None, 1.5),
        plot_extension="png",
        fit_rim=True)

    normalized_K = K / radius**2
    normalized_rim = fit["rim_height"] / radius
    normalized_floor = fit["floor_elevation"] / radius

    save_morphometric_parameters(
        filename=output_csv,
        crater=crater,
        popt_map=fit,
        psig_map=uncertainty,
        isnew=i == 0,
        additional_data={
            "K": K,
            "K_over_R2": normalized_K,
            "rim_height_over_R": normalized_rim,
            "floor_elevation_over_R": normalized_floor})
    print(f"K = {K:.0f} m^2")

df = pd.read_csv(output_csv)
K_normalized = df["K_over_R2"]
rim_height = df["rim_height_over_R"]
rim_height_std = df["rim_height_std"] / radius
floor_elevation = df["floor_elevation_over_R"]
floor_elevation_std = df["floor_elevation_std"] / radius
floor_radius = df["floor_radius"] / radius
floor_radius_std = df["floor_radius_std"] / radius

plt.figure()
plt.errorbar(
    K_normalized,
    rim_height,
    yerr=rim_height_std,
    fmt="o", capsize=1,
    color="hotpink", markerfacecolor="none", markersize=2)

plt.title(f"Rim height vs. K — initial rim = {rim_height_initial:.0f} m")
plt.xlabel(r"Normalized degradation state $K/R^2$")
plt.ylabel(r"Normalized rim height $h_{\mathrm{rim}}/R$")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    ROOT / "plots" / f"rim_height_vs_K_rim_{int(rim_height_initial)}m_002.png",
    dpi=300)
plt.close()

plt.figure()
plt.errorbar(
    K_normalized,
    floor_elevation,
    yerr=floor_elevation_std, fmt="o",
    capsize=1, color="darkviolet",
    markerfacecolor="none", markersize=2)
plt.title(
    f"Floor elevation vs. K — initial rim = {rim_height_initial:.0f} m")
plt.xlabel(r"Normalized degradation state $K/R^2$")
plt.ylabel(r"Normalized floor elevation $h_{\mathrm{floor}}/R$")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    ROOT / "plots" / f"floor_elevation_vs_K_rim_{int(rim_height_initial)}m_002.png",
    dpi=300)
plt.close()

plt.figure()
plt.errorbar(
    K_normalized, floor_radius,
    yerr=floor_radius_std,
    fmt="o", capsize=1,
    color="cyan", markerfacecolor="none", markersize=2)

plt.title(
    f"Floor radius vs. K — initial rim = {rim_height_initial:.0f} m")
plt.xlabel(r"Normalized degradation state $K/R^2$")
plt.ylabel(r"Normalized floor radius $r_{\mathrm{floor}}/R$")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    ROOT / "plots" / f"floor_radius_vs_K_rim_{int(rim_height_initial)}m_002.png",
    dpi=300)
plt.close()