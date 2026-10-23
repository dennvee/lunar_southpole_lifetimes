from pathlib import Path
from cratermaker import Simulation
from cratermaker.utils.general_utils import format_large_units
from morphometry import (
    get_morphometry_from_binned_profile,
    save_morphometric_parameters)

scriptdir = Path(__file__).resolve().parent
rootdir = scriptdir.parent
datadir = rootdir / "data_Kvst"
simrootdir = rootdir / "simdata_Kvst"
plotdir = rootdir / "plots_Kvst"
diameter = 1e3

diam_str = (
    format_large_units(diameter, quantity="length").replace(".", "_").replace(" ", "_")
)
simdir = simrootdir / f"time_as_K_D{diam_str}"

local_radius = diameter
pix = local_radius / 250.0
sim = Simulation(
    simdir=simdir,
    surface="hireslocal",
    morphology="basicmoon",
    pix=pix,
    local_radius=local_radius,
    do_counting=False,
    ask_overwrite=False,
    reset=True,
    rng_seed=1128302)

# This is our test crater, scaled to take up most of the local region.
crater = sim.emplace(
    name=diam_str,
    diameter=diameter,
    location=sim.surface.local.location,
    monte_carlo=False,  # Use nominal values for the test crater
)[0]

original_diameter = crater.diameter
original_radius = crater.radius
original_rim_height = crater.rim_height
original_floor_elevation = crater.floor_elevation

def morphometric_fit(interval, **kwargs):
    """
    This is defined as a function that can be passed as a save action to the Simulation object.
    """
    fit_vars = [
        "radius",
        "floor_radius",
        "wall_curvature",
        "floor_blend",
        "rim_width",
        "rim_height",
        "frac_ejrim",
        "ejprofile",
    ]
    plot_vars = ["radius", "floor_elevation", "floor_radius", "rim_height", "rim_width"]

    ymin = 1.1 * original_floor_elevation
    ymax = 2.5 * original_rim_height

    time = sim.time if sim.time is not None else 0.0
    time_str = format_large_units(time, "time")

    popt_map, psig_map = get_morphometry_from_binned_profile(
        sim=sim,
        crater=crater,
        fit_range=(None, 1.25 * original_radius / crater.radius),
        plotfile=plotdir / f"{sim.simdir.stem}_{interval:03}.png",
        fit_vars=fit_vars,
        ymin=ymin,
        ymax=ymax,
        titletext=f"Time: {time_str}",
        save_intermediate=False,
        plot_vars=plot_vars,
    )

    additional_data = {"time": time}
    save_morphometric_parameters(
        filename=datadir / f"{sim.simdir.stem}.csv",
        crater=crater,
        popt_map=popt_map,
        psig_map=psig_map,
        isnew=interval == 0,
        additional_data=additional_data,
    )


sim.add_save_action({morphometric_fit: {}})

sim.save()

# Save the initial parameters of the test crater for posterity
sim.counting.to_csv_file(
    crater_type="emplaced",
    craters=[crater],
    interval=0,
    output_file=datadir / f"{simdir.stem}_original_values.csv",
)

sim.run(age=2000.0, time_interval=10.0)
