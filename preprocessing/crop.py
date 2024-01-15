import argparse
import os
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from tqdm import tqdm

def crop_and_save_tile(input_file, output_folder, tile_size=2009):
    with rasterio.open(input_file) as src:
        profile = src.profile
        width, height = src.width, src.height

        for i in range(0, width - tile_size, tile_size):
            for j in range(0, height - tile_size, tile_size):
                window = Window(i, j, min(tile_size, width - i), min(tile_size, height - j))
                tile_data = src.read(window=window, resampling=Resampling.bilinear)

                tile_profile = profile.copy()
                tile_profile.update({
                    'width': tile_size,
                    'height': tile_size,
                    'transform': rasterio.windows.transform(window, src.transform)
                })

                filename = os.path.splitext(os.path.basename(input_file))[0]
                output_filename = os.path.join(output_folder, f"tile_{filename}_{i}_{j}.tif")

                with rasterio.open(output_filename, 'w', **tile_profile) as dst:
                    dst.write(tile_data)

def crop_tif_files(input_folder, output_folder, tile_size=2009):
    os.makedirs(output_folder, exist_ok=True)

    tif_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

    for tif_file in tqdm(tif_files, desc="Cropping tiles into smaller tiles"):
        input_path = os.path.join(input_folder, tif_file)
        crop_and_save_tile(input_path, output_folder, tile_size)


def main(tif_folder, output, tile_size):
    crop_tif_files(tif_folder, output, tile_size)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tif-folder')
    parser.add_argument("--output", default="./split_folder/", help="Output folder for splits")
    parser.add_argument("--tile-size", default=2009, help="Size of tiles")
    args = parser.parse_args()
    
    
    tif_folder = args.tif_folder
    output = args.output
    tile_size = args.tile_size
    main(tif_folder, output, tile_size)
