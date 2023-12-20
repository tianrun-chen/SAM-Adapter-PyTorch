import numpy as np
import matplotlib.pyplot as plt
import os

folderPath = 'test/numpy'
for numpyFile in os.listdir(folderPath):
    maskArray = np.load(f'{folderPath}/{numpyFile}')
    maskArray *= 255.0
    plt.imsave(f'greyscale/{numpyFile}.png',maskArray)
    print(maskArray.min())
    print(maskArray.max())