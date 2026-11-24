from cratermaker import Simulation

sim = Simulation(
    surface="hireslocal", local_location=(0.0, -89.9999), local_radius=150.0e3, pix=80.0, ask_overwrite=False, reset=True
)
sim.update_elevation("/scratch/negishi/dvalade/lunar_southpole_lifetimes/Barker_LOLA_tifs/LDEM_80S_80MPP_ADJ.TIF")
sim.plot(show=False, save=True)

'''sim = Simulation(
    surface="datasurface", local_location=(0.0, -89.9999), local_radius=150.0e3, ask_overwrite=False, reset=True,
dem_file_list= ["/scratch/negishi/dvalade/lunar_southpole_lifetimes/Barker_LOLA_tifs/LDEM_80S_80MPP_ADJ.TIF"])  #add dem_file_list=[filename] to use your own DEM
sim.plot(show=False, save=True) '''

''' # Alternatively you could do this to have more control over resolution
sim = Simulation(
    surface="hireslocal", local_location=(0.0, -89.9999), local_radius=150.0e3, pix=1e3, ask_overwrite=False, reset=True
)
sim.update_elevation("path/to/dem/file")
sim.plot(show=True, save=True)'''
