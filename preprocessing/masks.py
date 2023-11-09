import numpy as np
from matplotlib.path import Path as PolygonPath
from pathlib import Path
from tqdm import tqdm

from typing import List, Tuple

from load_munich_ma import LoadMunich
from bounding_box_ma import BoundingBox

IMAGE_SIZES = {
    'munich_florian':(2500, 2500),
    'munich_tepe':(2500, 2500)
}


class MaskMaker:
    """This class looks for all files defined in the metadata, and
    produces masks for all of the .tif files saved there.
    These files will be saved in <org_folder>_mask/<org_filename>.npy

    Attributes:
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
    """

    def __init__(self, geojson_path, images_path, city, data_folder: Path = Path('data')) -> None:
        self.data_folder = data_folder
        self.loader = LoadMunich(images_path,geojson_path,city)
        self.polygon_images, self.polygon_pixels = self.loader.run()

    def process(self) -> None:

        for city, files in self.polygon_images.items():
            print(f'Processing {city}')
            # first, we make sure the mask file exists; if not,
            # we make it
            masked_city = self.data_folder / f"{city}_masks"
            x_size, y_size = IMAGE_SIZES[city]
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
            x_size, y_size = IMAGE_SIZES[city]
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