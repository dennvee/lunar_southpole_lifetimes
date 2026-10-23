from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import cratermaker as cm
from morphometry import get_morphometry_from_binned_profile as mbp

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_FILE = (
    PROJECT_ROOT
    / "catalogue"
    / "CraterHostsforPSRuniques_v2_reprojected.csv")

DEM_10MPP = (
    PROJECT_ROOT
    / "Barker_LOLA_tifs"
    / "LDEM_83S_10MPP_ADJ.TIF")

DEM_20MPP = (
    PROJECT_ROOT
    / "Barker_LOLA_tifs"
    / "LDEM_80S_20MPP_ADJ.TIF")

OUTPUT_DIR = PROJECT_ROOT / "C_outputs_PSR_morphometry_003"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "PSR_crater_morphometry_003.csv"

crater_names =[
"HNP_743572_0381610",
"HNP_788526_1860292",
"HNP_812903_0803994",
"HNP_815044_3495692",
"HNP_820222_1668813",
"HNP_804181_1811748",
"HNP_806076_3265429",
"HNP_809329_0825134",
"HNP_811972_0750187",
"HNP_812903_0803994",
"HNP_816358_3521018",
"HNP_817587_1956239",
"HNP_819514_3064871",
"HNP_820222_1668813",
"HNP_820835_3523443",
"HNP_821862_3337421",
"HNP_814862_0839133",
"HNP_815044_3495692",
"HNP_819514_3064871",
"HNP_820222_1668813",
"HNP_820835_3523443",
"HNP_829762_3270088",
"HNP_835365_2984849",
"HNP_835577_2977728",
"HNP_838282_2911064",
"HNP_838693_1427787",
"HNP_839402_1723932",
"HNP_840549_3039489",
"HNP_840906_2655700",
"HNP_843343_2242153"
]

''' first batch of craters
crater_names = [
    "HNP_876263_0930110",
    "HNP_877836_0822028",
    "HNP_771598_2061917",
    "HNP_877794_0859925",
    "HNP_877965_1262388",
    "HNP_877723_0833390",
    "HNP_877279_2413214",
    "HNP_876730_1473716",
    "HNP_876363_1384995",
    "10-1-079344",
    "HNP_878009_1614990",
    "HNP_877936_0926713",
    "HNP_877743_2865020",
    "HNP_877661_3056012",
    "HNP_877604_1102984",
    "HNP_877521_0969561",
    "HNP_877253_1684281", 
    "HNP_877228_1049050",
    "HNP_877029_0029402",
    "HNP_876821_2012272",
    "HNP_876721_0097184",
    "HNP_876609_0914578",
    "HNP_876347_0918827",
    "HNP_876310_0943579",
    "HNP_876078_0091867",
    "HNP_875498_0908216",
    "HNP_875147_0219934"
    ]'''
   
fit_vars = [
    "radius",
    "rim_height",
    "rim_width",
    "floor_radius",
    "wall_curvature",
    "floor_blend",
    "frac_ejrim",
    "ejprofile",]

df = pd.read_csv(CSV_FILE)
results = []

for count, crater_name in enumerate(crater_names, start=1):

    try:
        match = df[
            (df["CRATER_ID"].astype(str) == crater_name)
            | (df["PSR_NAME"].astype(str) == crater_name)
        ]

        if len(match) == 0:
            print(f"\n[WARNING] Crater not found: {crater_name}")
            continue

        if len(match) > 1:
            print(f"\n[WARNING] Multiple matches found: {crater_name}")
            continue

        row = match.iloc[0]

        diameter_m = float(row["diam_m"])
        radius_m = diameter_m / 2.0
        lon = float(row["longitude"])
        lat = float(row["latitude"])

        dem_file = DEM_10MPP if lat <= -83.0 else DEM_20MPP
        simdir = OUTPUT_DIR / f"simdata_{crater_name}"

        print(f"\nProcessing {count} of {len(crater_names)}")
        print(f"Crater: {crater_name}")
        print(f"Location: ({lon}, {lat})")
        print(f"Diameter: {diameter_m:.1f} m")
        print(f"DEM: {dem_file.name}")

        sim = cm.Simulation(
            surface="datasurface",
            local_location=(lon, lat),
            local_radius=2.0 * diameter_m,
            simdir=str(simdir),
            reset=True,
            ask_overwrite=False,
            dem_file_list=[str(dem_file)])

        crater = sim.Crater.maker(
            name=crater_name,
            diameter=diameter_m,
            location=(lon, lat))

        fit, uncertainty = mbp(
            sim=sim,
            crater=crater,
            fit_vars=fit_vars,
            nrbins=100,
            fit_range=(None, 1.5),
            plot_extension="png",
            fit_rim=True )

        fitted_radius = fit["radius"]

        results.append({
            "CRATER_ID": row["CRATER_ID"],
            "PSR_NAME": row["PSR_NAME"],
            "longitude": lon,
            "latitude": lat,

            "catalogue_diameter_m": diameter_m,
            "catalogue_radius_m": radius_m,

            "fitted_radius_m": fit["radius"],
            "fitted_radius_std": uncertainty["radius"],

            "rim_height_m": fit["rim_height"],
            "rim_height_std": uncertainty["rim_height"],
            "rim_height_over_R": fit["rim_height"] / fitted_radius,

            "rim_width_m": fit["rim_width"],
            "rim_width_std": uncertainty["rim_width"],

            "floor_elevation_m": fit["floor_elevation"],
            "floor_elevation_std": uncertainty["floor_elevation"],
            "floor_elevation_over_R": (
                fit["floor_elevation"] / fitted_radius
            ),

            "floor_radius_m": fit["floor_radius"],
            "floor_radius_std": uncertainty["floor_radius"],
            "floor_radius_over_R": (
                fit["floor_radius"] / fitted_radius
            ),

            "wall_curvature": fit["wall_curvature"],
            "wall_curvature_std": uncertainty["wall_curvature"],

            "floor_blend": fit["floor_blend"],
            "floor_blend_std": uncertainty["floor_blend"],

            "frac_ejrim": fit["frac_ejrim"],
            "frac_ejrim_std": uncertainty["frac_ejrim"],

            "ejprofile": fit["ejprofile"],
            "ejprofile_std": uncertainty["ejprofile"]
        })

        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        plt.close("all")

        print(f"Fitted radius: {fit['radius']:.2f} m")
        print(f"Rim height: {fit['rim_height']:.2f} m")
        print(f"Floor elevation: {fit['floor_elevation']:.2f} m")
        print(f"Floor radius: {fit['floor_radius']:.2f} m")
        print(f"Saved results to: {OUTPUT_CSV}")

    except Exception as error:
        plt.close("all")
        print(f"\n[WARNING] Failed crater: {crater_name}")
        print(f"[WARNING] Error: {error}")
        print("Continuing to next crater...")

print(f"\nFinished. Successfully fitted {len(results)} craters.")
print(f"Results saved to: {OUTPUT_CSV}")