from pathlib import Path
import sys
import cratermaker as cm
ROOT = Path(__file__).resolve().parents[1]
rng_seed = int(sys.argv[1])
simdir = ROOT / "simulation_directories" / f"lunar_sp_seed_{rng_seed:03d}"

local_location = (0.0, -90.0)
local_radius_m = 150000
pix = 750

run_age_My = 4310
time_interval_My = 10
input_csv = ROOT / "southpole_inputcsv" / "southpole_basins_psrs_inputsv2.csv"
largest_crater_m = 200000

sim = cm.Simulation(
    target="Moon",
    surface="hireslocal",
    morphology="realmoon",
    rng_seed=rng_seed,
    local_location=local_location,
    local_radius=local_radius_m,
    pix=pix,
    do_counting=False,
    simdir=simdir,
    reset=True,
    ask_overwrite=False,
    quasimc_file=str(input_csv))

sim.largest_crater = largest_crater_m
sim.run(age=run_age_My, time_interval=time_interval_My)