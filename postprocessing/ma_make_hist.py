import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


folderPath = 'test/numpy'
for numpyFile in os.listdir(folderPath):
    maskArray = np.load(f'{folderPath}/{numpyFile}')
    plt.hist(maskArray)
    plt.savefig(f'hists/{numpyFile}.png')