import os
import requests
from tqdm import tqdm

# in the making

def utm_to_download(x: str, y: str):
    """ Converts UTM coordinates to download-link for open data."""

    zone = 32

    x_tile = x[:-3]
    y_tile = y[:-3]

    return f'https://download1.bayernwolke.de/a/dop40/data/{zone}{x_tile}_{y_tile}.tif'


def download_open_data(x: str, y: str, out_dir: str):
    """ Downloads open data from Bavarian government. """

    url = utm_to_download(x, y)
    filename = url.split('/')[-1]

    out_file = os.path.join(out_dir, filename)

    if not os.path.exists(out_file):
        r = requests.get(url, allow_redirects=True)
        open(out_file, 'wb').write(r.content)
    else:
        print(f'File {out_file} already exists, skipping download.')


def calculate_needed_tiles(x1: str, y1: str, x2: str, y2: str):
    """ Calculates the tiles needed to cover the area of interest. """
    tiles = []

    # y values
    for y_val in range(int(y1[:-3]), int(y2[:-3])+1):
        # x values
        for x_val in range(int(x1[:-3]), int(x2[:-3])+1):
            tiles.append((f'{x_val}000', f'{y_val}000'))
    
    return tiles

def download_list_of_tiles(tiles: list, out_dir: str):
    """ Downloads a list of tiles. """
    for tile in tqdm(tiles, desc='Downloading tiles'):
        download_open_data(tile[0], tile[1], out_dir)

def calculate_and_download(x1: str, y1: str, x2: str, y2: str, out_dir: str):
    """ Calculates and downloads all tiles needed to cover the area of interest. """
    tiles = calculate_needed_tiles(x1, y1, x2, y2)
    download_list_of_tiles(tiles, out_dir)