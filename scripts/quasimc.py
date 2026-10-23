from pathlib import Path
import sys
import cratermaker as cm
 
simdir= "southpole_qmc"
local_location = (0.0, 90.0)
local_radius_m = 150000
pix = 750

run_age_My = 4310
time_interval_My = 10
input_csv = "southpole_basins_psrs_inputs.csv"
largest_crater_m = 200000

sim = cm.Simulation(
    target="Moon",
    surface="hireslocal",
    morphology="realmoon",
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