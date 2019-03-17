import fastai
from fastai.vision import *
from fastai.callbacks import *

from torchvision.models import vgg16_bn


path_hr = os.getcwd() + '\\img_512_HE\\'
path_lr = os.getcwd() + '\\img_64_LR\\'

il = ImageItemList.from_folder(path_hr)


def resize_one(fn,i):
    dest = path_lr/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, 64, use_min=True)
    img = img.resize(targ_sz, resample=PIL.Image.LANCZOS).convert('RGB')
    img.save(dest, quality=60)

if __name__ == '__main__':
    parallel(resize_one, il.items)

