from pathlib import Path
import imageio.v2 as imageio

image_dir = (
    Path(__file__).resolve().parent
    /"sim_directories"
    / "lunar_qmc_oman_test_008"
    / "surface_images")

output_file = (
    Path(__file__).resolve().parent
    / "lunar_oman_test008.mp4")

images = sorted(image_dir.glob("local_surface_hillshade*.png"))

with imageio.get_writer(
    output_file,
    fps=10,
    codec="libx264") as writer:
    for image_file in images:
        writer.append_data(imageio.imread(image_file))