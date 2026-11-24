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

DEM_240MPP = (
    PROJECT_ROOT
    / "Barker_LOLA_tifs"
    / "LDEM_60S_240MPP_ADJ.TIF")

OUTPUT_DIR = PROJECT_ROOT / "simulation_directories"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = PROJECT_ROOT / "data" / "real_crater_morphometry_localpsr_test.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "plots" / "binned_profiles_localpsr").mkdir(parents=True, exist_ok=True)

crater_names = [ 
"HNP_754199_1480595",
"HNP_748843_2120819",
"10-2-007973",
"10-2-003032",
"10-2-002276",
"10-2-001898",
"10-2-001784",
"10-2-001656",
"10-1-096961",
"10-1-000851",
"10-1-000876",
"10-2-001488",
"10-2-001557",
"10-1-000838",
"10-1-000916",
"10-1-001007",
"10-1-090536",
"10-2-002068",
"HNP_873241_2761777",
"HNP_873186_1906545",
"HNP_873056_0628731",
"HNP_873015_1315378",
"HNP_872790_2758489",
"HNP_872710_1296321",
"HNP_872396_1386545",
"HNP_872369_2620971",
"HNP_872297_1021763",
"HNP_871968_1109466",
"HNP_871763_2708413",
"HNP_871736_2269161",
"HNP_871733_1373653",
"HNP_870984_1088641",
"HNP_870757_1352440",

"10-2-001602_124259",
"10-2-001667_53360",
"10-2-001784_245",
"10-2-001880_121682",
"10-2-002276_399",
"10-2-003032_513",
"10-2-004499_1559",
"10-2-004672_896",
"10-2-010355_42704",
"HNP_748843_2120819",
"HNP_754199_1480595",
"HNP_776376_2804694",
"HNP_787910_1058143",
"HNP_790379_0710497",
"HNP_793754_2319556",
"HNP_803298_2645101",
"HNP_814120_3061047",
"HNP_825168_0999817",
"HNP_831228_0942543",
"HNP_835206_1719940",
"HNP_836631_1738453",
"HNP_838163_3362681",
"HNP_844062_0223354",
"HNP_844139_2357454",
"HNP_844437_1604770",
"HNP_846334_0177556",
"HNP_846550_1021673",
"HNP_847280_2299446",
"HNP_847552_1898435",
"HNP_848386_1483531",
"HNP_848600_0582213",
"HNP_848838_0738836",
"HNP_849406_2242994",
"HNP_850660_3318176",
"HNP_851879_0572727",
"HNP_851987_3073026",
"HNP_852016_1527212",
"HNP_852241_2890742",
"HNP_852663_1480965",
"HNP_852793_1856572"
"HNP_853177_1049666",
"HNP_853765_2099637",
"HNP_854158_2440932",
"HNP_854752_2814300",
"HNP_855501_2377155",
"HNP_855622_2114568",
"HNP_856308_2159603",
"HNP_856518_3239151",
"HNP_856624_1827423",
"HNP_856763_2398668",
"HNP_856814_1868991"
"HNP_857721_2139922",
"HNP_857930_2719566",
"HNP_858392_2177552",
"HNP_858648_2474502",
"HNP_859708_2891283",
"HNP_859820_1056427",
"HNP_860299_2190570",
"HNP_860806_1037338",
"HNP_860985_2159477",
"HNP_861061_1384423",
"HNP_861223_0910048",
"HNP_861660_1151556",
"HNP_862198_2343763",
"HNP_862315_2767583",
"HNP_862550_2557350",
"HNP_862882_2455357",
"HNP_863098_1685838",
"HNP_863226_1834056",
"HNP_863506_1038286",
"HNP_864282_3548261",
"HNP_864565_1095931"
]

#crater_names= ["10-1-000838" ]#, #shoemaker
         #      "HNP_854024_0840253", #faustini
            #   "10-1-001007" #haworth
           #    ] 

#crater_names= ["10-1-000838"]
#100-200m craters
'''
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
"HNP_843343_2242153"]
'''
''' first batch of craters 1km-2km craters
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
        if diameter_m < 1000:
            dem_file = DEM_10MPP
        elif diameter_m > 10000:
            dem_file = DEM_240MPP
        else:
            dem_file = DEM_20MPP
        #dem_file= DEM_240MPP
        #dem_file = DEM_10MPP if lat <= -83.0 else DEM_20MPP
        simdir = OUTPUT_DIR / f"simdata_{crater_name}"

        print(f"\nProcessing {count} of {len(crater_names)}")
        print(f"Crater: {crater_name}")
        print(f"Location: ({lon}, {lat})")
        print(f"Diameter: {diameter_m:.1f} m")
        print(f"DEM: {dem_file.name}")

        sim = cm.Simulation(
            surface="datasurface",
            local_location=(lon, lat),
            local_radius=0.5*diameter_m,
            simdir=str(simdir),
            reset=True,
            ask_overwrite=False,
            dem_file_list=[str(dem_file)])

        crater = sim.Crater.maker(
            name=crater_name,
            diameter=diameter_m,
            location=(lon, lat),
            peak_height=0)
       
        crater._rings= []
        fit, uncertainty = mbp(
            sim=sim,crater=crater,
            fit_vars=fit_vars,
            nrbins=100,
            fit_range=(None, 1.5),
            plot_extension="png",
            fit_rim=True,
            save_intermediate= True )

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
                fit["floor_elevation"] / fitted_radius),
            "floor_radius_m": fit["floor_radius"],
            "floor_radius_std": uncertainty["floor_radius"],
            "floor_radius_over_R": (
                fit["floor_radius"] / fitted_radius),
            "wall_curvature": fit["wall_curvature"],
            "wall_curvature_std": uncertainty["wall_curvature"],
            "floor_blend": fit["floor_blend"],
            "floor_blend_std": uncertainty["floor_blend"],
            "frac_ejrim": fit["frac_ejrim"],
            "frac_ejrim_std": uncertainty["frac_ejrim"],
            "ejprofile": fit["ejprofile"],
            "ejprofile_std": uncertainty["ejprofile"]})
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    except Exception as error:
        print(f"[WARNING] Failed {crater_name}: {error}")
    finally:
        plt.close("all")
    print(f"Successfully fitted {len(results)} craters.")