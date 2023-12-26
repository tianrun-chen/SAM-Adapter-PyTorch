import os
import argparse
import geopandas as gpd
import rasterio
import rasterio.transform
import rasterio.features
import rasterio.shutil
import numpy as np
import shapely.ops
from shapely.geometry import Polygon
from tqdm import tqdm

# Generates Masks and pairs them with their respective images
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

    def unify_crs(self, mask, image):
        mask = mask.to_crs(image.crs)
        return mask
    
    def create_mask_geometry(self, mask):
        return gpd.GeoSeries(data=mask["geometry"], crs=mask.crs)
    
    def paired_name(self, image_path):
        return image_path.split(os.sep)[-1].split(".")[0]
    
    def save(self, image, unioned, paired_name):
        
        width, height = image.width, image.height
        mask = np.zeros((height, width), dtype=np.uint8)
        minx, miny, maxx, maxy = image.bounds
        transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
        mask = rasterio.features.geometry_mask([unioned], transform=transform, invert=False, out_shape=(height, width))
        
        np.save(os.path.join(self.output_mask_folder,paired_name), mask)
        rasterio.shutil.copy(image, os.path.join(self.output_image_folder,paired_name), driver='GTiff')
    
    def process(self):
        
        image_paths = self.get_image_paths()
        geojson_paths = self.get_geojson_paths()

        os.makedirs(self.output_image_folder, exist_ok=True)
        os.makedirs(self.output_mask_folder, exist_ok=True)

        for image_path in tqdm(image_paths, desc="Processing images", position=0):

            with rasterio.open(image_path) as src:
                for geojson_path in tqdm(geojson_paths, desc="Generating masks and pairing them with their respective images", leave=False, position=1):
                    
                    bbox = src.bounds
                    mask = gpd.read_file(geojson_path, bbox=bbox)

                    if mask.empty:
                        continue
                    
                    mask = self.unify_crs(mask, src)

                    mask_geom = self.create_mask_geometry(mask)
        
                    unioned = shapely.ops.unary_union(mask_geom)
                    self.save(image=src, unioned=unioned, paired_name=self.paired_name(image_path))


def main():
    parser = argparse.ArgumentParser(description="Generate masks from images based on GeoJSON polygons.")
    parser.add_argument("--tif-folder", help="Path to the folder containing TIFF images.")
    parser.add_argument("--geojson-folder", help="Path to the folder containing GeoJSON files.")
    parser.add_argument("--output-image-folder", default= "images_preprocessed_non_split",help="Path to the output folder for processed images.")
    parser.add_argument("--output-mask-folder", default= "masks_preprocessed_non_split",help="Path to the output folder for generated masks.")
    args = parser.parse_args()

    masker = Masker(args.tif_folder, args.geojson_folder, args.output_image_folder, args.output_mask_folder)
    masker.process()

if __name__ == "__main__":
    main()
