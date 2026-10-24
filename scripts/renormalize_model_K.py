from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
#data_dir = Path(__file__).resolve().parent.parent / "morphometryv2"
rim_heights = [35, 40, 50, 60]

for rim_height in rim_heights:
    input_csv = ROOT/ "data" / f"morphometry_vs_K_rim_{rim_height}m.csv"
    output_csv = ROOT/ "data" / f"morphometry_vs_K_rim_{rim_height}m_renormalized.csv"

    df = pd.read_csv(input_csv)
    R = df["radius"]
    df["K_over_R2"] = df["K"] / R**2
    df["rim_height_over_R"] = df["rim_height"] / R
    df["rim_height_std_over_R"] = df["rim_height_std"] / R
    df["floor_elevation_over_R"] = df["floor_elevation"] / R
    df["floor_elevation_std_over_R"] = df["floor_elevation_std"] / R
    df["floor_radius_over_R"] = df["floor_radius"] / R
    df["floor_radius_std_over_R"] = df["floor_radius_std"] / R
    df.to_csv(output_csv, index=False)
    print(f"{rim_height} m: saved {output_csv}")