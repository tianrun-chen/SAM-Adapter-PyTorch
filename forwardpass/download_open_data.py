import os
import requests
from tqdm import tqdm
from pyproj import Transformer


class DownloadOpenData:
    def utm_to_download(self, x: str, y: str):
        """ Converts UTM coordinates to download-link for open data."""

        zone = 32

        x_tile = x[:-3]
        y_tile = y[:-3]

        return f'https://download1.bayernwolke.de/a/dop40/data/{zone}{x_tile}_{y_tile}.tif'


    def download_open_data(self, x: str, y: str, out_dir: str):
        """ Downloads open data from Bavarian government. """

        url = self.utm_to_download(x, y)
        filename = url.split('/')[-1]

        out_file = os.path.join(out_dir, filename)

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)        

        if not os.path.exists(out_file):
            r = requests.get(url, allow_redirects=True)
            open(out_file, 'wb').write(r.content)
        else:
            print(f'File {out_file} already exists, skipping download.')


    def calculate_needed_tiles(self, x1: str, y1: str, x2: str, y2: str):
        """ Calculates the tiles needed to cover the area of interest. """
        tiles = []

        # y values
        for y_val in range(int(y1[:-3]), int(y2[:-3])+1):
            # x values
            for x_val in range(int(x1[:-3]), int(x2[:-3])+1):
                tiles.append((f'{x_val}000', f'{y_val}000'))

        return tiles

    def download_list_of_tiles(self, tiles: list, out_dir: str):
        """ Downloads a list of tiles. """
        for tile in tqdm(tiles, desc='Downloading tiles'):
            self.download_open_data(tile[0], tile[1], out_dir)

    def calculate_and_download(self, x1: str, y1: str, x2: str, y2: str, out_dir: str):
        """ Calculates and downloads all tiles needed to cover the area of interest. """
        tiles = self.calculate_needed_tiles(x1, y1, x2, y2)
        self.download_list_of_tiles(tiles, out_dir)

    def trafo_wgs84_etrs89(self, lat: float, long: float):
        """ Transforms coordinates from WGS84 to ETRS89. """
        return Transformer.from_crs("epsg:4326", "epsg:25832").transform(lat, long)

    def wgs84_download(self, x1: float, y1: float, x2: float, y2: float, out_dir: str):
        """ input: wgs84 coos, Downloads open data from Bavarian government. """
        x1, y1 = self.trafo_wgs84_etrs89(x1, y1)
        x2, y2 = self.trafo_wgs84_etrs89(x2, y2)
        self.calculate_and_download(str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), out_dir)