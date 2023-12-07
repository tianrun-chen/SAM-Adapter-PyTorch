import os
import requests

# in the making

def utm_to_download(x: str, y: str):
    """ Converts UTM coordinates to download-link for open data."""

    zone = 32

    x_tile = x[:3]
    y_tile = y[:4]

    return f'https://download1.bayernwolke.de/a/dop40/data/{zone}{x_tile}_{y_tile}.tif'


def download_open_data(x: str, y: str, out_dir: str):
    """ Downloads open data from Bavarian government. """

    url = utm_to_download(x, y)
    filename = url.split('/')[-1]

    out_file = os.path.join(out_dir, filename)

    if not os.path.exists(out_file):
        print(f'Downloading {url} to {out_file}')
        r = requests.get(url, allow_redirects=True)
        open(filename, 'wb').write(r.content)
    else:
        print(f'File {out_file} already exists, skipping download.')


download_open_data('688025', '5330033', '/')