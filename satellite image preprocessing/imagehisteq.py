import sys
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import scipy
import skimage
from skimage import data, color, io, exposure


directory = os.getcwd() + '\\img_512_final\\'
pathlist = Path(directory).iterdir()


for path in pathlist:

    imageload = io.imread(str(path))

    y_eq = skimage.img_as_float32(exposure.equalize_hist(imageload))

    filename = str(os.path.basename(path))[:str(os.path.basename(path)).find(".")]

    skimage.io.imsave('C:\\Users\\rohit\\Workspace\\SuperresolutionModels\\unet\\img_512_HE\\' + filename + '_HE.png', y_eq, quality = 100)



