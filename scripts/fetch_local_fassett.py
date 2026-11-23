from pathlib import Path
import argparse
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cratermaker import Crater, Simulation

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Fetch crater-centered DEMs within 150 km of the lunar south pole.")
parser.add_argument("min_diameter", type=float)
parser.add_argument("max_diameter", type=float)
parser.add_argument("--radius-km", type=float, default=150)
parser.add_argument("--dem", type=Path, default=ROOT.parent / "Lunar_Morphometry_Revisited" / "playground" / "Barker_LOLA_tifs" / "LDEM_60S_240MPP_ADJ.TIF")
parser.add_argument("--csv", type=Path, default=ROOT / "catalogues" / "CraterHostsforPSRuniques_v2_reprojected.csv")
parser.add_argument("--extent", type=float, default=2, help="Local DEM radius in crater radii (default: 4).")
args = parser.parse_args()

if not 0 < args.min_diameter < args.max_diameter or args.radius_km <= 0 or args.extent <= 0:
    parser.error("Require 0 < min_diameter < max_diameter, positive radius and extent.")
for path in (args.csv, args.dem):
    if not path.is_file():
        parser.error(f"File not found: {path}")

name = f"D{args.min_diameter:g}_{args.max_diameter:g}m"
out_dir = ROOT / "simulation_directories" / "fassett_local" / name
png_dir = ROOT / "plots" / "fassett_local_dems" / name
out_dir.mkdir(parents=True, exist_ok=True)
png_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(args.csv)
for col in ("diam_m", "latitude", "longitude"):
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Great-circle distance from the south pole depends only on latitude.
df["distance_km"] = 1737.4 * np.deg2rad(90 + df["latitude"])
df = df.loc[
    df["diam_m"].ge(args.min_diameter)
    & df["diam_m"].lt(args.max_diameter)
    & df["latitude"].between(-90, 90)
    & df["longitude"].notna()
    & df["distance_km"].le(args.radius_km)
].copy()

print(f"Selected {len(df)} craters within {args.radius_km:g} km of the south pole")
print(f"Diameter range: [{args.min_diameter:g}, {args.max_diameter:g}) m")
print(f"DEM: {args.dem}")
print(f"Images: {png_dir}")

for i, (idx, row) in enumerate(df.iterrows(), 1):
    crater_name = next((str(row[c]) for c in ("CRATER_ID", "PSR_NAME") if c in df and pd.notna(row[c]) and str(row[c]).strip()), f"crater_{idx}")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in crater_name)
    simdir = out_dir / f"simdata_{safe_name}_{idx}"
    outfile = png_dir / f"{safe_name}_{idx}_{row['diam_m'] / 1000:.3f}km.png"
    try:
        crater = Crater.maker(name=crater_name, diameter=float(row["diam_m"]), location=(float(row["longitude"]), float(row["latitude"])))
        print(f"[{i}/{len(df)}] {crater_name}: D={row['diam_m']:g} m, pole distance={row['distance_km']:.1f} km", flush=True)
        sim = Simulation(
            surface="datasurface",
            local_location=crater.location,
            local_radius=crater.radius * args.extent,
            simdir=str(simdir),
            reset=True,
            ask_overwrite=False,
            dem_file_list=[str(args.dem)],
        )
        sim.plot(
            plot_style="hillshade",
            variable_name="face_elevation",
            save=True,
            savepath=str(outfile),
            show=False,
            include_counting=False,
            close_when_done=True,
        )
        source = simdir / "surface_images" / "local_surface_hillshade_face_elevation000000.png"
        if source.exists():
            shutil.copy2(source, outfile)
        print(f"Saved: {outfile}" if outfile.exists() else f"Warning: no PNG for {crater_name}")
    except Exception as error:
        print(f"Warning: {crater_name} failed: {error}")
    finally:
        plt.close("all")