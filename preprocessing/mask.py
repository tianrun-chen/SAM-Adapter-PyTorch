import os
import argparse
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import xarray as xr

class Masker:
    def __init__(self, tif_folder, geojson_folder, output_image_folder, output_mask_folder):
        self.tif_folder = tif_folder
        self.geojson_folder = geojson_folder
        self.output_image_folder = output_image_folder
        self.output_mask_folder = output_mask_folder

    def get_image_paths(self):
        tif_files = [f for f in os.listdir(self.tif_folder) if f.endswith('.tif')]
        return [os.path.join(self.tif_folder, f) for f in tif_files]

    def get_geojson_paths(self):
        geojson_files = [f for f in os.listdir(self.geojson_folder) if f.endswith('.geojson')]
        return [os.path.join(self.geojson_folder, f) for f in geojson_files]

    def read_polygon_data(self, geojson_path):
        return gpd.read_file(geojson_path)

    def check_polygons_intersect(self, image_bounds, polygon_data):
        converted = xr.
        return polygon_data[polygon_data.geometry.intersects(image_bounds)]

    def clip_image(self, src, intersecting_polygons):
        return mask(src, intersecting_polygons.geometry, crop=True)

    def save_clipped_image(self, output_image_path, clipped, src_profile):
        with rasterio.open(output_image_path, 'w', **src_profile) as dst:
            dst.write(clipped)

    def create_binary_mask(self, clipped):
        mask_array = np.zeros(clipped.shape, dtype=np.uint8)
        mask_array[clipped.mask] = 1
        return mask_array

    def save_mask(self, mask_output_path, mask_array):
        np.save(mask_output_path, mask_array)

    def process_images(self):
        image_paths = self.get_image_paths()
        geojson_paths = self.get_geojson_paths()

        os.makedirs(self.output_image_folder, exist_ok=True)
        os.makedirs(self.output_mask_folder, exist_ok=True)

        for geojson_path in geojson_paths:
            polygon_data = self.read_polygon_data(geojson_path)

            for image_path in image_paths:
                with rasterio.open(image_path) as src:
                    image_bounds = src.bounds
                    intersecting_polygons = self.check_polygons_intersect(image_bounds, polygon_data)

                    if intersecting_polygons.empty:
                        print(f"No intersecting polygons found for {image_path}. Skipping...")
                        continue

                    clipped, _ = self.clip_image(src, intersecting_polygons)

                    # Save the clipped image
                    output_image_path = os.path.join(self.output_image_folder, os.path.basename(image_path))
                    self.save_clipped_image(output_image_path, clipped, src.profile)

                    # Save the mask
                    mask_array = self.create_binary_mask(clipped)
                    mask_output_path = os.path.join(
                        self.output_mask_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_mask.npy"
                    )
                    self.save_mask(mask_output_path, mask_array)

def main():
    parser = argparse.ArgumentParser(description="Process images and generate masks based on GeoJSON polygons.")
    parser.add_argument("--tif_folder", help="Path to the folder containing TIFF images.")
    parser.add_argument("--geojson_folder", help="Path to the folder containing GeoJSON files.")
    parser.add_argument("--output_image_folder", help="Path to the output folder for processed images.")
    parser.add_argument("--output_mask_folder", help="Path to the output folder for generated masks.")
    args = parser.parse_args()

    masker = Masker(args.tif_folder, args.geojson_folder, args.output_image_folder, args.output_mask_folder)
    masker.process_images()

if __name__ == "__main__":
    main()
