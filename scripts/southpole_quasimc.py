from pathlib import Path
import argparse
import cratermaker as cm

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("rng_seed", type=int)
parser.add_argument("--csv", default="southpole_basins_psrs_inputsv1.csv")
parser.add_argument("--pix", type=int, default=650)
args = parser.parse_args()

rng_seed = args.rng_seed
pix = args.pix
input_csv = ROOT / "southpole_inputcsv" / args.csv
simdir = ROOT / "simulation_directories" / f"southpole_{rng_seed:03d}"

local_location = (0.0, -89.99)
local_radius_m = 150000
run_age_My = 4310
time_interval_My = 10
largest_crater_m = 200000

sim = cm.Simulation(
    target="Moon",
    surface="hireslocal",
    morphology="basicmoon",
    rng_seed=rng_seed,
    local_location=local_location,
    local_radius=local_radius_m,
    pix=pix,
    do_counting=False,
    simdir=simdir,
    reset=True,
    ask_overwrite=False,
    quasimc_file=input_csv)

sim.largest_crater = largest_crater_m
sim.run(age=run_age_My, time_interval=time_interval_My)