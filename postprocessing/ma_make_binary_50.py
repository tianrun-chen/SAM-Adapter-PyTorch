import numpy as np
from PIL import Image
import os

folderPath = 'test/numpy'
for numpyFile in os.listdir(folderPath):
    maskArray = np.load(f'{folderPath}/{numpyFile}')
    #threshold = maskArray.mean()
    threshold = maskArray.min()+(maskArray.max()-maskArray.min())*0.9
    binMask = maskArray > threshold
    img = Image.fromarray(binMask)
    split_filename = numpyFile.split('.')
    img.save(f'binary/{split_filename[0]}.png')
    #plt.imshow(maskArray)
    #plt.savefig(f'binary/{numpyFile}.png')