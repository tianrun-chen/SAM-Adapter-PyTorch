import numpy as np
from matplotlib.path import Path as PolygonPath
from pathlib import Path
from tqdm import tqdm

from typing import List, Tuple

from load_munich_ma import LoadMunich
from bounding_box_ma import BoundingBox
import argparse
import os

class MaskMaker:
    """This class looks for all files defined in the metadata, and
    produces masks for all of the .tif files saved there.
    These files will be saved in <org_folder>_mask/<org_filename>.npy

    Attributes:
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
    """

    def __init__(self, geojson_path, images_path, city, image_size, utm_size, data_folder: Path = Path('data')) -> None:
        self.data_folder = data_folder
        self.image_size = image_size
        self.utm_size = utm_size
        self.loader = LoadMunich(images_path,geojson_path,city, self.image_size, self.utm_size)
        self.polygon_images, self.polygon_pixels = self.loader.run()

    def process(self) -> None:

        for city, files in self.polygon_images.items():
            print(f'Processing {city}')
            # first, we make sure the mask file exists; if not,
            # we make it
            masked_city = self.data_folder / f"{city}_masks"
            x_size, y_size = self.image_size
            if not masked_city.exists(): masked_city.mkdir()

            for image, polygons in tqdm(files.items()):
                mask = np.zeros((x_size, y_size))
                for polygon in polygons:
                    mask += self.make_mask(self.polygon_pixels[polygon], (x_size, y_size))

                mask[mask > 1] = 1
                split_image = image.split('.')
                image_name = split_image[0]
                np.save(masked_city / f"{image_name}.npy", mask)

    def process_bounding_box(self):
        boundingbox = BoundingBox()
        boxes = boundingbox.buildFromPolygon(self.polygon_pixels)
        edges = boundingbox.getEdges(boxes)

        for city, files in self.polygon_images.items():
            print(f'Processing Boundingbox for {city}')
            # first, we make sure the mask file exists; if not,
            # we make it
            masked_city = self.data_folder / f"{city}_bounding_masks"
            x_size, y_size = self.image_size
            if not masked_city.exists(): masked_city.mkdir()

            for image, polygons in tqdm(files.items()):
                mask = np.zeros((x_size, y_size))
                for polygon in polygons:
                    mask += self.make_mask(edges[polygon], (x_size, y_size))

                mask[mask > 1] = 1

                split_image = image.split('.')
                image_name = split_image[0]
                np.save(masked_city / f"{image_name}.npy", mask)




    @staticmethod
    def make_mask(coords: List, imsizes: Tuple[int, int]) -> np.array:
        """https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
        """
        poly_path = PolygonPath(coords)

        x_size, y_size = imsizes
        x, y = np.mgrid[:x_size, :y_size]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        mask = poly_path.contains_points(coors)

        return mask.reshape(x_size, y_size).astype(float)
    

def main():
    parser = argparse.ArgumentParser(description='Produce masks by processing geojson and image files.')
    parser.add_argument('--image-folder', type=str, help='Path to the folder containing input images')
    parser.add_argument('--geojson-folder', type=str, help='Path to the folder where geojson data is saved')
    #parser.add_argument('--output-folder-images', type=str, help='Path to the folder where images will be saved')
    #parser.add_argument('--output-folder-masks', type=str, help='Path to the folder where masks will be saved')
    parser.add_argument('--city', type=str, default='munich', help='Name of the city')
    parser.add_argument('--image-size', type=int, help='Size of the images')
    parser.add_argument('--utm-size', type=int, help='Size of the utm tiles')

    args = parser.parse_args()

    image_folder = args.image_folder
    geojson_folder = args.geojson_folder
    #output_folder_images = args.output_folder_images
    #output_folder_masks = args.output_folder_masks
    city = args.city
    image_size = args.image_size
    utm_size = args.utm_size

    for geojson_file in os.listdir(geojson_folder):
            maskmaker = MaskMaker(os.path.join(geojson_folder,geojson_file), image_folder, city, (image_size, image_size), (utm_size, utm_size))
            maskmaker.process()

if __name__ == "__main__":
    main()
