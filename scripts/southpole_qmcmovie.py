from pathlib import Path
import sys
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
rng_seed = int(sys.argv[1])

image_dir = ROOT / "simulation_directories" / f"lunar_sp_seed_{rng_seed:03d}" / "surface_images"
output_dir = ROOT / "plots" / "southpole_movies"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"lunar_sp_seed_{rng_seed:03d}.mp4"

images = sorted(image_dir.glob("local_surface_hillshade*.png"))
if not images:
    raise FileNotFoundError(f"No hillshade frames found in {image_dir}")

with imageio.get_writer(output_file, fps=10, codec="libx264") as writer:
    for image_file in images:
        writer.append_data(imageio.imread(image_file))

print(f"Saved movie: {output_file}")