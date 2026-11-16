import cratermaker as cm

simdir = "simdata-1_2"

# Note, that for these examples we pass ask_overwrite=False and reset=True to the Simulation constructor. This will suppress
# prompts that ask the user if they want to overwrite existing files, which would would prevent these examples from running on their
# own when building the documentation pages. Alternatively, calling cm.cleanup(simdir) will remove all pre-existing output files.


# Initialize a quick Moon simulation. We will reduce the resolution to gridlevel 6 and turn off counting to speed up the simulation for this example.
sim = cm.Simulation(target="Moon", gridlevel=6, do_counting=False, simdir=simdir, ask_overwrite=False, reset=True)

sim.run(age=4310)
sim.show3d(variable_name="face_elevation", cmap="Greys_r")