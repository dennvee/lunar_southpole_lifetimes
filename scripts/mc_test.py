from pathlib import Path
import sys
import cratermaker as cm
ROOT = Path(__file__).resolve().parents[1]
simdir = ROOT / "simulation_directories" / f"mc_test5"

local_location = (0.0, -89.999)
local_radius_m = 150000
pix = 750

run_age_My = 1000
time_interval_My = 10
largest_crater_m = 200000
rng_seed= 248
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
    ask_overwrite=False)

sim.largest_crater = largest_crater_m
sim.run(age=run_age_My, time_interval=time_interval_My)