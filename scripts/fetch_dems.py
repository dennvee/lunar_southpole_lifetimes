from pathlib import Path
import sys
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from cratermaker import Crater, Simulation

min_diameter_m = float(sys.argv[1])
max_diameter_m = float(sys.argv[2])

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_FILE = (
    PROJECT_ROOT
    / "catalogues"
    / "CraterHostsforPSRuniques_v2_reprojected.csv")

DEM_10MPP = (
    PROJECT_ROOT
    / "Barker_LOLA_tifs"
    / "LDEM_83S_10MPP_ADJ.TIF")

DEM_20MPP = (
    PROJECT_ROOT
    / "Barker_LOLA_tifs"
    / "LDEM_80S_20MPP_ADJ.TIF")

OUTPUT_DIR = (
    PROJECT_ROOT
    / "B_PSR_DEMs_100m_200m"
    / f"D{int(min_diameter_m)}_{int(max_diameter_m)}m")

PNG_DIR = OUTPUT_DIR / "PNGs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_FILE)
df = df[
    (df["diam_m"] >= min_diameter_m)
    & (df["diam_m"] < max_diameter_m)].copy()

print(f"Diameter range: {min_diameter_m}–{max_diameter_m} m")
print(f"Number of craters: {len(df)}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"PNG directory: {PNG_DIR}")

for count, (_, row) in enumerate(df.iterrows(), start=1):
    crater_name = "UNKNOWN"

    try:
        if pd.notna(row["CRATER_ID"]):
            crater_name = str(row["CRATER_ID"])
        else:
            crater_name = str(row["PSR_NAME"])

        diameter_m = float(row["diam_m"])
        lon = float(row["longitude"])
        lat = float(row["latitude"])

        dem_file = DEM_10MPP if lat <= -83.0 else DEM_20MPP

        safe_name = (
            crater_name
            .replace(" ", "_")
            .replace("/", "_"))

        crater = Crater.maker(
            name=crater_name,
            diameter=diameter_m,
            location=(lon, lat))

        simdir = OUTPUT_DIR / f"simdata_{safe_name}"

        outfile = (
            PNG_DIR
            / f"{safe_name}_{diameter_m / 1000.0:.2f}km.png")

        print(f"\nProcessing {count} of {len(df)}")
        print(f"Crater: {crater_name}")
        print(f"Location: ({lon}, {lat})")
        print(f"Diameter: {diameter_m / 1000.0:.3f} km")
        print(f"DEM: {dem_file.name}")

        sim = Simulation(
            surface="datasurface",
            local_location=crater.location,
            local_radius=crater.radius * 4.0,
            simdir=str(simdir),
            reset=True,
            ask_overwrite=False,
            dem_file_list=[str(dem_file)])

        sim.plot(
            plot_style="hillshade",
            variable_name="face_elevation",
            save=True,
            savepath=str(outfile),
            show=False,
            include_counting=False,
            close_when_done=True)

        source_png = (
            simdir
            / "surface_images"
            / "local_surface_hillshade_face_elevation000000.png")

        if source_png.exists():
            shutil.copy2(source_png, outfile)
            print(f"Copied: {outfile}")
        else:
            print(f"[WARNING] Hillshade PNG not found: {source_png}")

        plt.close("all")

    except Exception as error:
        plt.close("all")
        print(f"[WARNING] Failed crater {count}: {crater_name}")
        print(f"[WARNING] Error: {error}")
        print("Continuing to next crater...")