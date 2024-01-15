import os
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box
import argparse
import shutil
from tqdm import tqdm

def check_geojson_and_tif_intersection(geojson_file, tif_file, output_folder):
    # Read GeoJSON file
    gdf = gpd.read_file(geojson_file)

    # Open TIFF file to get its bounding box
    with rasterio.open(tif_file) as src:
        tif_bbox = src.bounds

    # Create a bounding box polygon from the TIFF bounding box
    tif_polygon = gpd.GeoSeries([box(*tif_bbox)], crs=gdf.crs)

    # Check intersection
    intersection = gdf.intersects(tif_polygon.unary_union)

    if any(intersection):
        output_tif_path = os.path.join(output_folder, os.path.basename(tif_file))
        shutil.copy(tif_file, output_tif_path)

def save_intersecting_files(tif_folder, geojson_folder, output_folder):
    tif_files = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]
    geojson_files = [f for f in os.listdir(geojson_folder) if f.endswith('.geojson')]

    os.makedirs(output_folder, exist_ok=True)

    for tif_file in tqdm(tif_files, desc="Removing files with no defined geometries"):
        for geojson_file in geojson_files:
            tif_path = os.path.join(tif_folder, tif_file)
            geojson_path = os.path.join(geojson_folder, geojson_file)
            check_geojson_and_tif_intersection(geojson_path, tif_path, output_folder)


def main(tif_folder, geojson_folder, output):
    save_intersecting_files(tif_folder, geojson_folder, output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geojson-folder')
    parser.add_argument('--tif-folder')
    parser.add_argument("--output", default="./split_folder_empty_removed/", help="Output folder for splits with removed empty tiles")
    args = parser.parse_args()

    geojson_folder = args.geojson_folder
    tif_folder = args.tif_folder
    output = args.output
    main(tif_folder, geojson_folder, output)
