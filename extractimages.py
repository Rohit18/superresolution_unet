import os
import tifffile as tiff
import numpy as np
import slidingwindow as sw
from pathlib import Path
from matplotlib import pyplot as plt
import scipy
from PIL import Image
import skimage
import gdal




directory = os.getcwd() + '\\additionaldata\\'
pathlist = Path(directory).iterdir()

#print(directory)

for path in pathlist:

    print(path)
    
    img = np.array(scipy.ndimage.imread(str(path)))

    windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, 512, 0.2)

    for i, window in enumerate(windows):
        subset = img[ window.indices() ]
        filename = str(os.path.basename(path))[:str(os.path.basename(path)).find(".")]
        skimage.io.imsave('C:\\Users\\rohit\\Workspace\\SuperresolutionModels\\unet\\img_512\\' + filename + '_' + str(i) + '.png', subset)


