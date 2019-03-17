import os
import tifffile as tiff
import numpy as np
import slidingwindow as sw
from pathlib import Path
from matplotlib import pyplot as plt
import scipy
from PIL import Image
import skimage


directory = os.getcwd() + '\\img_512\\'
pathlist = Path(directory).iterdir()


for path in pathlist:
    
    img = np.array(plt.imread(str(path)))
    filename = str(os.path.basename(path))[:str(os.path.basename(path)).find(".")]

    count =  np.count_nonzero(img)

    if count == 786432:
        skimage.io.imsave('C:\\Users\\rohit\\Workspace\\SuperresolutionModels\\unet\\img_512_final\\' + filename + '.png', img)

